import os


def main() -> None:
    import uvicorn

    # Localhost by default: the app has no auth. Containers set CKVIZ_HOST=0.0.0.0 and publish on 127.0.0.1.
    uvicorn.run("ckviz.api:app", host=os.environ.get("CKVIZ_HOST", "127.0.0.1"),
                port=int(os.environ.get("CKVIZ_LISTEN_PORT", "8000")))
