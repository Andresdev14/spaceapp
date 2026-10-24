import { createHash } from "node:crypto";

// A SpaceApp image as the release workflow publishes it (BuildKit sbom + provenance mode=max), with real digests, and a
// fake ghcr.io serving it. `provenanceFor` changes one platform's provenance statement.
const sha = (bytes) => `sha256:${createHash("sha256").update(bytes).digest("hex")}`;

export function attestedImage({
  repository = "oll4com/spaceapp-core",
  architectures = ["amd64", "arm64"],
  workflowSha = "f1c4e8351bb899ab9bd741b6e0d531b6ba8436c8",
  runId = "39996418646",
  provenanceFor = (statement) => statement
} = {}) {
  const objects = new Map();
  const put = (value) => {
    const bytes = Buffer.from(JSON.stringify(value));
    const digest = sha(bytes);
    objects.set(digest, bytes);
    return { digest, size: bytes.length };
  };
  const manifests = [];
  const attestationDescriptors = [];
  for (const architecture of architectures) {
    const image = put({ schemaVersion: 2, mediaType: "application/vnd.oci.image.manifest.v1+json", architecture });
    manifests.push({ mediaType: "application/vnd.oci.image.manifest.v1+json", ...image, platform: { os: "linux", architecture } });
    const provenance = put(provenanceFor({
      _type: "https://in-toto.io/Statement/v0.1",
      predicateType: "https://slsa.dev/provenance/v1",
      subject: [{ name: `pkg:docker/ghcr.io/${repository}@1.1.8?platform=linux%2F${architecture}`, digest: { sha256: image.digest.slice(7) } }],
      predicate: {
        buildDefinition: { internalParameters: {
          github_repository: "oll4com/spaceapp",
          github_workflow_ref: "oll4com/spaceapp/.github/workflows/release.yml@refs/heads/main",
          github_workflow_sha: workflowSha,
          github_run_id: runId
        } },
        runDetails: { builder: { id: `https://github.com/oll4com/spaceapp/actions/runs/${runId}/attempts/1` } }
      }
    }, architecture));
    const attestation = put({ schemaVersion: 2, layers: [
      { ...put({ spdxVersion: "SPDX-2.3" }), annotations: { "in-toto.io/predicate-type": "https://spdx.dev/Document" } },
      { ...provenance, annotations: { "in-toto.io/predicate-type": "https://slsa.dev/provenance/v1" } }
    ] });
    attestationDescriptors.push({ mediaType: "application/vnd.oci.image.manifest.v1+json", ...attestation, platform: { os: "unknown", architecture: "unknown" },
      annotations: { "vnd.docker.reference.digest": image.digest, "vnd.docker.reference.type": "attestation-manifest" } });
  }
  const index = { schemaVersion: 2, mediaType: "application/vnd.oci.image.index.v1+json", manifests: [...manifests, ...attestationDescriptors] };
  const { digest } = put(index);
  return { repository, digest, index, objects };
}

/** A fake ghcr.io for `images`; `requests` lists every path read, `tags` maps a tag to an image's index digest. */
export function fakeRegistry(images, { tags = {}, tamper = new Set() } = {}) {
  const requests = [];
  const fetch = async (url) => {
    const { pathname, searchParams } = new URL(url);
    requests.push(`${pathname}${searchParams.size ? `?${searchParams}` : ""}`);
    if (pathname === "/token") return Response.json({ token: "anonymous" });
    const [, , owner, name, kind, reference] = pathname.split("/");
    const image = images.find((candidate) => candidate.repository === `${owner}/${name}`);
    const digest = tags[`${owner}/${name}:${reference}`] ?? reference;
    const bytes = image?.objects.get(digest);
    if (!bytes || !["manifests", "blobs"].includes(kind)) return new Response("not found", { status: 404 });
    return new Response(tamper.has(digest) ? Buffer.concat([bytes, Buffer.from(" ")]) : bytes);
  };
  return { fetch, requests };
}
