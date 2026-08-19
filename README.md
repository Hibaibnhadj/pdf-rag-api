# PDF RAG API

A FastAPI service for question-answering over PDF documents using retrieval-augmented generation (RAG).

## Architecture

The API is a FastAPI service. Single endpoint: `POST /ask`, expects `{"pdf_path": "...", "question": "..."}`, returns `{"answer": "..."}`.

**Request pipeline:**

The embedding model is loaded once at application startup and kept in memory for the pod's lifetime.

**In-memory embedding cache:** chunk embeddings for a given PDF are cached in memory, keyed by (file path, last-modified time). A second question about an already-seen PDF skips re-reading, re-chunking, and re-embedding entirely. See "Performance & load testing" for measured impact. Known limitation: this cache lives in the pod's RAM only, is lost on every pod restart, and does not help concurrent first-time requests (see below).

## Deployment (OpenShift)

The app is deployed on OpenShift (`hiba1-dev` project) using:

| Resource | Name | Purpose |
|---|---|---|
| BuildConfig | `pdf-rag-api` | Builds the container image from this Git repo (Docker strategy) |
| ImageStream | `pdf-rag-api` | Stores the built image |
| DeploymentConfig | `pdf-rag-api` | Runs the container; ConfigChange + ImageChange triggers mean it automatically redeploys whenever a new image is built or the config changes |
| Service | `pdf-rag-api` | Internal networking, port 8000 |
| Route | `pdf-rag-api` | Public HTTP endpoint |

**Secrets:** the LLM endpoint URL (`MODEL_URL`), previously hardcoded in `app.py`, is now stored in an OpenShift Secret (`pdf-rag-api-secrets`, type Opaque, key `MODEL_URL`) and injected into the DeploymentConfig as an environment variable. The app reads it at startup via `os.environ.get("MODEL_URL")`. No sensitive values remain in source code or on GitHub. The bearer token used to authenticate to the LLM endpoint is handled separately via the pod's Kubernetes service-account token (`/var/run/secrets/kubernetes.io/serviceaccount/token`), a native, already-secure mechanism.

**Container runtime note:** OpenShift runs containers as a non-root, randomly assigned UID. Anything written to disk at runtime (e.g. NLTK's tokenizer data) must go to a writable path, hence `NLTK_DATA=/tmp/nltk_data` and `HOME=/tmp`. Writing to the app's working directory (e.g. `/app`) fails with a PermissionError.

**Known operational note:** the OpenShift Developer Sandbox auto-scales idle deployments to 0 replicas after a period of inactivity. If the API appears down, check DeploymentConfigs -> pdf-rag-api -> Details for the replica count and scale back to 1 if needed.

## CI/CD pipeline

The pipeline runs automatically on every git push to main. As of this writing it has three sequential Tekton tasks (updated from an earlier two-task version):

1. **clone-repo** - clones the repository into a shared workspace.
2. **run-tests** - installs dependencies and runs `pytest tests/ -v`.
3. **trigger-build** - if tests pass, runs `oc start-build pdf-rag-api --wait`, rebuilding the Docker image and pushing it to the internal registry. The DeploymentConfig's ImageChange trigger then automatically rolls out a new pod.

This closes a gap found during testing: earlier, a push would run tests but not rebuild/redeploy the image, the BuildConfig had to be triggered manually, meaning tested code could sit un-deployed for some time (in one case, about two weeks) before someone noticed and triggered a build by hand. With trigger-build in place and verified end-to-end, a successful push now results in tested code actually running in production with no manual step.

**Supporting Tekton resources:**

| Resource | Name | Purpose |
|---|---|---|
| Pipeline | `pdf-rag-api-pipeline` | The three tasks above |
| TriggerTemplate | `pdf-rag-api-trigger-template` | Defines the PipelineRun to create when triggered |
| TriggerBinding | `pdf-rag-api-trigger-binding` | Extracts the commit SHA from the GitHub push payload |
| EventListener | `pdf-rag-api-event-listener` | Receives webhook calls, starts a PipelineRun |
| Route | `pdf-rag-api-eventlistener` | Exposes the EventListener publicly so GitHub can reach it |

GitHub's repository webhook (Settings -> Webhooks) points at the EventListener's Route. On every push to main, GitHub calls that URL, the EventListener creates a new PipelineRun, and clone -> test -> build runs automatically.

**Manually re-running the pipeline** (e.g. to test without pushing): OpenShift Console -> Pipelines -> Pipelines -> pdf-rag-api-pipeline -> Actions -> Demarrer, choose the VolumeClaimTemplate for shared-workspace, and start.

**Design note - why an EventListener instead of a direct BuildConfig webhook:** a direct GitHub webhook pointed at the BuildConfig's Kubernetes API webhook URL is blocked on this cluster, the Developer Sandbox's RBAC policy denies anonymous external requests to the raw Kubernetes API (system:anonymous cannot create buildconfigs/webhooks). The same restriction blocks direct cluster-scope access to Tekton resources. Exposing a Tekton EventListener through an OpenShift Route sidesteps this: the EventListener is a normal application-level HTTP service, not a raw API call, so it isn't subject to the same RBAC restriction.

**Other environment quirks handled by the pipeline:**
- run-tests runs in a plain python:3.12-slim image, not the project's own Dockerfile, so NLTK setup is configured explicitly in the Tekton task.
- NLTK's SSRF protection blocks tokenizer downloads through this cluster's egress proxy unless NLTK_ALLOW_PROXIED_URLOPEN=1 is set.
- Each PipelineRun creates a PersistentVolumeClaim for its workspace. The Developer Sandbox limits a namespace to 10 PVCs total. Old, completed PipelineRuns should be deleted periodically (this cascades to delete their PVCs), otherwise new runs hang in Pending once the quota is hit. This cleanup is currently manual.

## Running the tests

Tests live in `tests/test_app.py` and use pytest. Current coverage (14 tests):
- **Chunking logic** - basic splitting, empty input, respecting max_chars, no content loss across chunk boundaries, a single sentence longer than the limit.
- **PDF reading** - correct text extraction from a real PDF, graceful failure on a missing file.
- **API endpoint (/ask)** - success case (LLM call mocked), missing-field validation, wrong-type validation, internal error propagation.
- **Integration test** - verifies the retrieval pipeline (chunk -> embed -> similarity search) actually selects the relevant chunk and passes correct content to the LLM prompt (only the external LLM call is mocked).

**Locally:**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
pip install pytest httpx
pytest tests/ -v
```

Note: app.py reads a Kubernetes service-account token at import time. Locally (outside a pod) this file doesn't exist, so the read is wrapped in try/except FileNotFoundError, falling back to token = "local-dev-token" so the app and tests can run outside OpenShift.

**Automatically:** just git push to main; check status under Pipelines -> PipelineRuns.

## Performance & load testing

The API was evaluated against PDFs of increasing size (5-81 pages, French text, each containing a unique marker fact to verify retrieval correctness), under the pod's 1 CPU / 1Gi memory limit.

**Sequential requests (one at a time):**

| Pages | Result | Response time |
|---|---|---|
| 5 | Success | ~5.4s |
| 15 | Success | ~11.6s |
| 30 | Success | ~14.6s |
| 50 | Success | ~19.7s |
| 65 | OOM killed | - |
| 81 | OOM killed | - |

Response time scales roughly linearly with page count. The memory ceiling under sequential (single-user) load sits between 50 and 65 pages.

**Embedding cache - measured effect** (30-page PDF, two consecutive questions):

| Call | Phase | Duration |
|---|---|---|
| 1 (cache miss) | read + chunk + embed | ~11.4s |
| 1 (cache miss) | retrieval + LLM call | ~11.7s |
| 1 (cache miss) | total | ~23.1s |
| 2 (cache hit) | retrieval + LLM call | ~1.7s |
| 2 (cache hit) | total | ~1.7s |

13.7x faster on the second question about the same document, the entire read/chunk/embed phase is eliminated.

**Concurrent requests:** 2 simultaneous requests against the same 30-page document (which succeeds fine sequentially) caused an OOM kill. Idle baseline memory (model loaded, no active request) was already at ~983.8 MiB out of the 1Gi limit; two overlapping embedding operations exceeded the remaining headroom almost immediately.

**Conclusion:** the API currently supports only one request at a time, reliably, regardless of document size. The embedding cache improves response time for repeated questions on an already-seen document, but does not solve the concurrency problem, two simultaneous first-time requests both miss the cache and both attempt full embedding in parallel.

## Vector DB - status and recommendation

Not implemented within the available time. Recommendation for a future iteration: a persistent vector store (e.g. ChromaDB) would let embeddings survive beyond a single pod's lifecycle. Whether it would also help the concurrency problem above is an open question, not yet validated, the current hypothesis is that it would not help two simultaneous first-time requests (both would still miss any cache/store and both would still attempt embedding in parallel), only repeated requests after a pod restart.

## Recommended next steps

1. Increase the pod's memory limit (e.g. 1Gi -> 2Gi), quick mitigation for large single documents.
2. Introduce a persistent vector store (see above).
3. Add request queuing or concurrency limiting on /ask.
4. Automate periodic cleanup of old PipelineRuns/PVCs (currently manual).

## Known limitations

- Embedding cache is in-memory only; lost on pod restart.
- No authentication or rate-limiting on /ask.
- Old PipelineRun/PVC cleanup is manual.
