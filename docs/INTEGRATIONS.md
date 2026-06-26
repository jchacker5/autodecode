# Agent integrations

autodecode's optimized MLX config is exposed as an OpenAI-compatible API. Start once, use everywhere.

## 1. Start the server

```bash
./start-mlx-server.sh
```

| Setting | Value |
|---------|-------|
| URL | `http://127.0.0.1:8080/v1` |
| Model ID | `default_model` |
| Temp | `0.05` |
| Prefill | `2048` |
| Thinking | disabled |

Health check: `curl http://127.0.0.1:8080/health`

---

## 2. OpenCode

Add to `~/.config/opencode/opencode.jsonc`:

```jsonc
{
  "model": "mlx/default_model",
  "provider": {
    "mlx": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "MLX (local)",
      "options": {
        "baseURL": "http://127.0.0.1:8080/v1",
        "apiKey": "mlx-local",
        "timeout": false
      },
      "models": {
        "default_model": {
          "name": "Ornith-9B MLX 4-bit",
          "tool_call": true,
          "limit": { "context": 32768, "output": 4096 }
        }
      }
    }
  }
}
```

```bash
opencode run -m mlx/default_model "your prompt"
```

---

## 3. Hermes

Add to `~/.hermes/config.yaml`:

```yaml
model:
  default: default_model
  provider: custom:mlx-ornith
  base_url: http://127.0.0.1:8080/v1
  context_length: 65536   # Hermes requires ≥64K for tool schemas

custom_providers:
  - name: mlx-ornith
    base_url: http://127.0.0.1:8080/v1
    models:
      default_model:
        context_length: 65536
    extra_body:
      chat_template_kwargs:
        enable_thinking: false

fallback_providers:
  - provider: xai-oauth
    model: grok-4.3
```

```bash
hermes -z "your prompt"
# or mid-session: /model custom:mlx-ornith:default_model
```

---

## After reboot

```bash
./start-mlx-server.sh
```

Both OpenCode and Hermes will reconnect automatically.