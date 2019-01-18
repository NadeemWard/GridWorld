import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gridworld",
    version="0.1",
    author="Nadeem Ward",
    author_email="patrick.ward@mail.mcgill.ca",
    description="A simple gridworld environment for reinforcement learning testing",
    long_description=long_description,
    url="https://github.com/NadeemWard/GridWorld",
    install_requires=['numpy'],
    packages=['gridworld'],
    license="MIT",
)