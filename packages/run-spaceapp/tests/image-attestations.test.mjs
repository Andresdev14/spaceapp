import test from "node:test";
import assert from "node:assert/strict";
import { mkdtemp, readFile, stat } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { evaluateImageAttestations, loadImageAttestations, verifyLockedImages } from "../src/image-attestations.mjs";
import { createDefaultConfig, readRecordedDigests, writeRuntimeFiles } from "../src/index.mjs";
import { attestedImage, fakeRegistry } from "./fixtures/attested-image.mjs";

const images = () => ["core", "cli", "browser"].map((target) => attestedImage({ repository: `oll4com/spaceapp-${target}` }));
const lockOf = (built) => ({ schemaVersion: 1, runtimeVersion: "1.1.8", images: Object.fromEntries(built.map((image) => [image.repository.split("-").pop(), image.digest])) });
const loader = (registry) => (repository, reference) => loadImageAttestations(repository, reference, { fetch: registry.fetch });

test("a published image passes with each platform's SBOM and provenance from the release workflow", async () => {
  const image = attestedImage();
  const registry = fakeRegistry([image]);
  const result = evaluateImageAttestations({ image: "core", ...await loadImageAttestations(image.repository, image.digest, { fetch: registry.fetch }) });
  assert.deepEqual(result.blockers, []);
  assert.deepEqual(Object.keys(result.platforms), ["amd64", "arm64"]);
  assert.deepEqual(result.platforms.amd64, { digest: image.index.manifests[0].digest, runId: "39996418646", workflowSha: "f1c4e8351bb899ab9bd741b6e0d531b6ba8436c8" });
  // The provenance statements are read; the SBOM documents (megabytes) are not.
  const sbomDigests = image.index.manifests.slice(2).map((descriptor) => JSON.parse(image.objects.get(descriptor.digest)).layers[0].digest);
  assert.ok(sbomDigests.every((digest) => !registry.requests.some((path) => path.endsWith(digest))));
  assert.equal(registry.requests.filter((path) => path.includes("/blobs/")).length, 2);
});

test("bytes that do not hash to the digest they were read by are refused", async () => {
  const image = attestedImage();
  const provenance = JSON.parse(image.objects.get(image.index.manifests[2].digest)).layers[1].digest;
  for (const tampered of [image.digest, image.index.manifests[2].digest, provenance]) {
    const registry = fakeRegistry([image], { tamper: new Set([tampered]) });
    await assert.rejects(loadImageAttestations(image.repository, image.digest, { fetch: registry.fetch }), /does not match its digest/);
  }
});

test("provenance about another image, from another workflow or without a commit is a blocker", () => {
  const cases = [
    [(statement) => ({ ...statement, subject: [{ digest: { sha256: "0".repeat(64) } }] }), /has no provenance about its linux\/amd64 image/],
    [(statement) => ({ ...statement, predicate: { ...statement.predicate, buildDefinition: { internalParameters: {
      ...statement.predicate.buildDefinition.internalParameters, github_workflow_ref: "oll4com/spaceapp/.github/workflows/release.yml@refs/heads/fork" } } } }), /was not built by oll4com\/spaceapp\/\.github\/workflows\/release\.yml@refs\/heads\/main/],
    [(statement) => ({ ...statement, predicate: { ...statement.predicate, runDetails: { builder: { id: "https://github.com/someone/spaceapp/actions/runs/39996418646/attempts/1" } } } }), /was not built by/],
    [(statement) => ({ ...statement, predicate: { ...statement.predicate, buildDefinition: { internalParameters: {
      ...statement.predicate.buildDefinition.internalParameters, github_workflow_sha: undefined } } } }), /names no source commit/]
  ];
  for (const [change, message] of cases) {
    const image = attestedImage({ provenanceFor: (statement, architecture) => architecture === "amd64" ? change(statement) : statement });
    const evidence = { index: image.index, attestations: [] };
    for (const descriptor of image.index.manifests.slice(2)) {
      const manifest = JSON.parse(image.objects.get(descriptor.digest));
      evidence.attestations.push({ descriptor, manifest, provenance: [JSON.parse(image.objects.get(manifest.layers[1].digest))] });
    }
    const result = evaluateImageAttestations({ image: "core", ...evidence });
    assert.equal(result.ok, false);
    assert.match(result.blockers.join(" "), message);
    assert.deepEqual(Object.keys(result.platforms), ["arm64"]);
  }
});

test("a locked package's images are verified once, before their digests are recorded or runtime.env names them", async () => {
  const built = images();
  const lock = lockOf(built);
  const registry = fakeRegistry(built);
  const verify = (imageLock) => verifyLockedImages(imageLock, { load: loader(registry), now: () => new Date("2026-10-24T06:00:00Z") });
  const root = await mkdtemp(join(tmpdir(), "spaceapp-image-attestations-"));
  await writeRuntimeFiles(root, createDefaultConfig({ version: "1.1.8" }), lock, verify);
  const evidence = JSON.parse(await readFile(join(root, "image-attestations.json"), "utf8"));
  assert.equal(evidence["1.1.8"].verifiedAt, "2026-10-24T06:00:00.000Z");
  assert.equal(evidence["1.1.8"].images.cli.index, lock.images.cli);
  assert.equal(evidence["1.1.8"].images.browser.platforms.arm64.workflowSha, "f1c4e8351bb899ab9bd741b6e0d531b6ba8436c8");
  assert.deepEqual(await readRecordedDigests(root), { "1.1.8": lock.images });
  // The next start with the same package reads nothing from the registry.
  const reads = registry.requests.length;
  await writeRuntimeFiles(root, createDefaultConfig({ version: "1.1.8" }), lock, verify);
  assert.equal(registry.requests.length, reads);

  // A package whose images do not check out writes nothing: no digests, no evidence, no runtime.env.
  const fresh = await mkdtemp(join(tmpdir(), "spaceapp-image-attestations-refused-"));
  const unattested = { ...lock, images: { ...lock.images, core: `sha256:${"9".repeat(64)}` } };
  await assert.rejects(writeRuntimeFiles(fresh, createDefaultConfig({ version: "1.1.8" }), unattested, verify),
    /SpaceApp 1\.1\.8 images could not be verified, so none of them was pulled: ghcr\.io\/oll4com\/spaceapp-core@sha256:9{64}: ghcr\.io answered HTTP 404/);
  for (const file of ["image-digests.json", "image-attestations.json", "runtime.env"]) {
    assert.equal(await stat(join(fresh, file)).catch(() => null), null, file);
  }
});
