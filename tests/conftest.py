import pytest
import subprocess
import time
import os

@pytest.fixture(scope="session", autouse=True)
def start_compose():
    # Make sure we run from the repo root
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    os.chdir(repo_root)

    # Build and start all services defined in docker-compose.yml under project 'nstapp'
    subprocess.run(
        ["docker-compose", "-p", "nstapp", "up", "-d", "--build"],
        check=True
    )

    # Wait for containers to finish initialization.
    # If your services need more than 5–10s, bump this up.
    time.sleep(10)

    yield

    # Tear down everything when tests complete
    subprocess.run(
        ["docker-compose", "-p", "nstapp", "down", "-v", "--remove-orphans"],
        check=True
    )
