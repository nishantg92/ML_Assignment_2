from pathlib import Path

from setuptools import find_packages, setup


ROOT_DIR = Path(__file__).resolve().parent


def get_requirements() -> list[str]:
    requirements_path = ROOT_DIR / "requirements.txt"
    return [
        requirement
        for line in requirements_path.read_text(encoding="utf-8").splitlines()
        if (requirement := line.strip()) and not requirement.startswith(("#", "-e ."))
    ]


setup(
    name="ml-model-for-assignment",
    version="0.0.1",
    author="Nishant G",
    author_email="2026aj05092@wilp.bits-pilani.ac.in",
    packages=find_packages(include=["src", "src.*"]),
    install_requires=get_requirements(),
    python_requires=">=3.11",
)
