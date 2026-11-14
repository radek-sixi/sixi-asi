# Running the engine as a local service

The engine can stay in the background and answer on the OpenAI API, the way a local model service does. One
command installs it as a systemd user service; no root is needed.

## Install once

```bash
tar xzf sixi-asi-*-jetson-thor-aarch64.tar.gz -C ~/.local/opt      # any directory you keep
cd ~/.local/opt/sixi-asi-*-jetson-thor-aarch64
./sixi setup          # checks the machine
./sixi pull           # the default model, 18.9 GB
./sixi link           # puts the command on the PATH as `sixi-asi` (a link in ~/.local/bin)
sixi-asi service install --parallel 4
```

`service install` writes `~/.config/systemd/user/sixi-asi.service` and starts nothing.

## Use

| Command | |
|---|---|
| `sixi-asi service start` | loads the model (8-15 s for Qwen3.8-27B) and serves `http://127.0.0.1:8089/v1` |
| `sixi-asi service status` | whether the service runs and whether the API answers (exit code 0 when it does) |
| `sixi-asi service logs` | the server log, `~/.sixi/service.log` |
| `sixi-asi service restart` | after `service install` with other options |
| `sixi-asi service stop` | unloads the model and frees its memory |
| `sixi-asi service remove` | stops the service and deletes the unit |

`sixi-asi chat` and `sixi-asi run "..."` talk to the running service instead of loading the model again.

## Clients

```bash
curl -s http://127.0.0.1:8089/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": "What is 17*23?"}], "max_tokens": 1024 }'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:8089/v1", api_key="none")
print(client.chat.completions.create(model="qwen3.8-27b",
      messages=[{"role": "user", "content": "Hello"}]).choices[0].message.content)
```

Programs that read the environment: `OPENAI_BASE_URL=http://127.0.0.1:8089/v1` and `OPENAI_API_KEY=none`.
The endpoints are `/v1/chat/completions` (also streamed), `/v1/models`, `/health` and `/metrics`; the request
fields are in [API.md](API.md). The `model` field of a request is not used to pick a model: the service answers
with the one it was installed with.

## Settings

Install again with other options, then `sixi-asi service restart`:

```bash
sixi-asi service install --parallel 8                       # more requests at once
sixi-asi service install --host 0.0.0.0 --api-key SECRET    # reachable from the network, with a key
sixi-asi service install -m qwen3.8-flash-next --parallel 1 # another model
sixi-asi service install --ctx 65536                        # longer context
sixi-asi service install --boot                             # start with your session
sixi-asi service install -- --spec-k-max 8                  # anything after -- goes to sixi-server
```

With `--boot` the service starts when you log in. To start it at boot without a login, also run
`loginctl enable-linger` once.

## What to expect

- **One model, loaded all the time.** The service does not load models on request and does not unload after
  idle time. With Qwen3.8-27B, `--parallel 4` and the default 32768-token context it holds 48 GB (44 GiB) of
  memory until `service stop`; a shorter context (`--ctx 8192`) or fewer parallel requests take less.
- **Port.** 8089 by default. If something else already listens there, the service keeps restarting and
  `service status` shows the last lines of the log; install with another `--port`.
- **A new version.** Unpack it next to the old one, run `./sixi link` and `sixi-asi service install` from the new
  directory, then `sixi-asi service restart`. Models stay in `~/.sixi/models`.
- **Without systemd** use `./sixi serve --parallel 4` in a terminal or under your own supervisor.
