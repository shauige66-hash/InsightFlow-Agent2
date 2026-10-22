# Deployment contracts

The default `compose.app.yaml` runs the API, PostgreSQL and Redis with loopback-only published ports. It supports SQL execution. Python/Plot also require the separate sandbox image and a trusted Docker runner; mounting only the socket is insufficient. PDF remains an optional browser capability and degrades to HTML/Markdown when unavailable.

On Windows/macOS Docker Desktop, run the API on the host and use `compose.yaml` for the database/cache. Existing host deployment remains supported.

For a trusted **Linux** all-container runner:

1. Set `INSIGHTFLOW_WORKSPACE` to the absolute Linux checkout path, for example `/srv/insightflow-agent2`. It must be the same path on the Docker daemon's host and inside API/worker. A remote daemon on another filesystem is unsupported.
2. Create that checkout's `outputs` and `uploads` directories and grant UID 10001 write access to these two directories. Keep application sources readable. Do not grant the sandbox access to the entire checkout.
3. Build the separate sandbox image on the host using `python -m sandbox.build_image`. Runtime does not implicitly pull/build it.
4. Set `DOCKER_GID` to the numeric group of the Docker socket. For rootless Docker, also set `INSIGHTFLOW_DOCKER_SOCKET` to the actual user socket path and use its permissions. API and worker need both the CLI (included in the API image) and access to this socket.
5. Start with both files: `docker compose -f deploy/compose.app.yaml -f deploy/compose.sandbox.yaml up -d --build`.
6. For the queue, also set `INSIGHTFLOW_EXECUTION=queue` and explicitly enable `--profile queue`. Setting the execution environment variable alone does not start a profiled worker. For example: `INSIGHTFLOW_EXECUTION=queue docker compose -f deploy/compose.app.yaml -f deploy/compose.sandbox.yaml --profile queue up -d --build`.

The Linux override does not change the default deployment's requirements. It mounts the checkout at the identical path so daemon-side `/sandbox`, `/in`, `/history`, `/out` and `/scratch` bind sources exist. Reports/uploads for this runner live in the checkout's directories, not the base image's unused `/app` volumes. The API/worker are trusted: access to the Docker socket gives substantial host control. The analysis container still receives only its selected input/source/attempt/history mounts, never the checkout or `.env`.

Before accepting a deployment, verify a CSV Python analysis and chart produce real sandbox invocations, a queue job is consumed by the worker, a long task renews its Run lease, and authenticated artifact links work. These deployment checks have not been executed by the offline repair task.

Run leases fence lifecycle and guarded model/tool/artifact submissions. They cannot undo a model request already in flight when a worker loses its lease; model execution remains at least once. `claimed` jobs are not automatically retried: inspect them before explicit requeue. Session pending dispatch is retried using the same client message ID, rather than inventing a second turn.

API keys enable owner-scoped Run/Turn/resource access. Authenticated callers may use their own uploaded resources or bundled example `data/`/`knowledge/` files. Trusted local mode with no API keys retains local file paths. Legacy ownerless Runs are assigned `local-demo` during schema upgrade, rather than guessed to belong to an arbitrary token owner.
