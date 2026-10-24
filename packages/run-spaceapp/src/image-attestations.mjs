import { createHash } from "node:crypto";
import { IMAGE_TARGETS } from "./image-lock.mjs";

// SpaceApp's own images carry BuildKit SBOM and SLSA provenance attestations (reviews EXT-C-F5, IR117-C-F7). The npm
// package pins each image index by digest, and the index lists one attestation manifest per platform image, so every
// object read here by digest is the shipped bytes or bound to them: no signature service is needed. A platform image
// passes when its provenance names it as the subject and oll4com/spaceapp's release workflow on main as the builder;
// the run and the public commit that built it are returned as evidence. The release (verify-published-containers.mjs)
// and the launcher at install time run this same check.
export const RELEASE_REPOSITORY = "oll4com/spaceapp";
export const RELEASE_WORKFLOW_REF = `${RELEASE_REPOSITORY}/.github/workflows/release.yml@refs/heads/main`;
const SBOM_PREDICATE = "https://spdx.dev/Document";
const PROVENANCE_PREDICATE = "https://slsa.dev/provenance/v1";
const REGISTRY = "https://ghcr.io";
const MANIFEST_TYPES = [
  "application/vnd.oci.image.index.v1+json",
  "application/vnd.oci.image.manifest.v1+json",
  "application/vnd.docker.distribution.manifest.list.v2+json",
  "application/vnd.docker.distribution.manifest.v2+json"
].join(",");
const MAX_OBJECT_BYTES = 4 * 1024 * 1024;
const READ_TIMEOUT_MS = 30_000;

const isAttestationOf = (entry, digest) =>
  entry?.annotations?.["vnd.docker.reference.type"] === "attestation-manifest" &&
  (digest === undefined || entry?.annotations?.["vnd.docker.reference.digest"] === digest);

/**
 * Checks one image index: every required linux platform (default: every one it lists) has an SBOM and a provenance
 * statement about that platform image, built by the release workflow, from `workflowSha` when given.
 */
export function evaluateImageAttestations({ image, index, attestations, architectures, workflowSha }) {
  const blockers = [];
  const platforms = {};
  const manifests = Array.isArray(index?.manifests) ? index.manifests : [];
  const required = architectures ?? manifests.filter((entry) => entry?.platform?.os === "linux").map((entry) => entry.platform.architecture);
  if (required.length === 0) blockers.push(`${image} lists no linux image.`);
  for (const architecture of required) {
    const platformManifest = manifests.find((entry) => entry?.platform?.os === "linux" && entry?.platform?.architecture === architecture);
    if (!platformManifest) {
      blockers.push(`${image} is missing linux/${architecture}.`);
      continue;
    }
    const matching = (Array.isArray(attestations) ? attestations : []).filter((entry) => isAttestationOf(entry?.descriptor, platformManifest.digest));
    const predicates = new Set(matching.flatMap((entry) => Array.isArray(entry?.manifest?.layers) ? entry.manifest.layers : [])
      .map((layer) => layer?.annotations?.["in-toto.io/predicate-type"]));
    if (!predicates.has(SBOM_PREDICATE)) blockers.push(`${image} does not expose an SBOM for linux/${architecture}.`);
    if (!predicates.has(PROVENANCE_PREDICATE)) {
      blockers.push(`${image} does not expose provenance for linux/${architecture}.`);
      continue;
    }
    const statement = matching.flatMap((entry) => Array.isArray(entry?.provenance) ? entry.provenance : []).find((candidate) =>
      candidate?.predicateType === PROVENANCE_PREDICATE &&
      (Array.isArray(candidate.subject) ? candidate.subject : []).some((subject) => `sha256:${subject?.digest?.sha256}` === platformManifest.digest));
    if (!statement) {
      blockers.push(`${image} has no provenance about its linux/${architecture} image ${platformManifest.digest}.`);
      continue;
    }
    const github = statement.predicate?.buildDefinition?.internalParameters ?? {};
    const builder = statement.predicate?.runDetails?.builder?.id;
    if (
      github.github_repository !== RELEASE_REPOSITORY ||
      github.github_workflow_ref !== RELEASE_WORKFLOW_REF ||
      !/^\d+$/.test(String(github.github_run_id ?? "")) ||
      typeof builder !== "string" ||
      !builder.startsWith(`https://github.com/${RELEASE_REPOSITORY}/actions/runs/${github.github_run_id}/`)
    ) {
      blockers.push(`${image} linux/${architecture} was not built by ${RELEASE_WORKFLOW_REF}.`);
    } else if (!/^[0-9a-f]{40}$/.test(String(github.github_workflow_sha ?? ""))) {
      blockers.push(`${image} linux/${architecture} provenance names no source commit.`);
    } else if (workflowSha && github.github_workflow_sha !== workflowSha) {
      blockers.push(`${image} linux/${architecture} was built from ${github.github_workflow_sha}, not ${workflowSha}.`);
    } else {
      platforms[architecture] = { digest: platformManifest.digest, runId: String(github.github_run_id), workflowSha: github.github_workflow_sha };
    }
  }
  return { ok: blockers.length === 0, blockers, platforms };
}

/**
 * Reads an image index (by digest or tag) and its attestation manifests and provenance statements from ghcr.io with an
 * anonymous pull token. Every object is hashed; one read by digest that does not hash to it is refused.
 */
export async function loadImageAttestations(repository, reference, { fetch = globalThis.fetch } = {}) {
  const tokenResponse = await fetch(`${REGISTRY}/token?scope=repository:${repository}:pull`, { signal: AbortSignal.timeout(READ_TIMEOUT_MS) });
  if (!tokenResponse.ok) throw new Error(`ghcr.io refused a read token for ${repository} (HTTP ${tokenResponse.status}).`);
  const { token } = await tokenResponse.json();
  const read = async (kind, ref) => {
    const response = await fetch(`${REGISTRY}/v2/${repository}/${kind}/${ref}`, {
      headers: { Authorization: `Bearer ${token}`, Accept: MANIFEST_TYPES },
      signal: AbortSignal.timeout(READ_TIMEOUT_MS)
    });
    if (!response.ok) throw new Error(`ghcr.io answered HTTP ${response.status} for ${repository} ${ref}.`);
    const bytes = Buffer.from(await response.arrayBuffer());
    if (bytes.length > MAX_OBJECT_BYTES) throw new Error(`${repository} ${ref} is larger than ${MAX_OBJECT_BYTES} bytes.`);
    const digest = `sha256:${createHash("sha256").update(bytes).digest("hex")}`;
    if (ref.startsWith("sha256:") && digest !== ref) throw new Error(`${repository}@${ref} does not match its digest.`);
    return { digest, json: JSON.parse(bytes.toString("utf8")) };
  };
  const { digest, json: index } = await read("manifests", reference);
  const attestations = [];
  for (const descriptor of Array.isArray(index?.manifests) ? index.manifests : []) {
    if (!isAttestationOf(descriptor) || typeof descriptor.digest !== "string") continue;
    const { json: manifest } = await read("manifests", descriptor.digest);
    const provenance = [];
    for (const layer of Array.isArray(manifest?.layers) ? manifest.layers : []) {
      if (layer?.annotations?.["in-toto.io/predicate-type"] === PROVENANCE_PREDICATE) provenance.push((await read("blobs", layer.digest)).json);
    }
    attestations.push({ descriptor, manifest, provenance });
  }
  return { digest, index, attestations };
}

/**
 * Install-time check of a package's image lock: each SpaceApp image must pass evaluateImageAttestations for every
 * platform its index lists. Returns the evidence an installation records (image-attestations.json); throws otherwise.
 */
export async function verifyLockedImages(lock, { load = loadImageAttestations, now = () => new Date() } = {}) {
  const images = {};
  const blockers = [];
  for (const target of IMAGE_TARGETS) {
    const repository = `${RELEASE_REPOSITORY}-${target}`;
    let result;
    try {
      result = evaluateImageAttestations({ image: `ghcr.io/${repository}@${lock.images[target]}`, ...await load(repository, lock.images[target]) });
    } catch (error) {
      result = { ok: false, blockers: [`ghcr.io/${repository}@${lock.images[target]}: ${error instanceof Error ? error.message : String(error)}`] };
    }
    if (result.ok) images[target] = { index: lock.images[target], platforms: result.platforms };
    else blockers.push(...result.blockers);
  }
  if (blockers.length > 0) {
    throw new Error(`SpaceApp ${lock.runtimeVersion} images could not be verified, so none of them was pulled: ${blockers.join(" ")}`);
  }
  return { runtimeVersion: lock.runtimeVersion, verifiedAt: now().toISOString(), images };
}
