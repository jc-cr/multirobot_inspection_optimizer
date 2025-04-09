# multirobot_inspection_optimizer

This project showcases a MILP solver for a ground and aerial robot fleet inspection route planning problem.

## Requirements

Docker and Docker compose recommended to run.
Otherwise python requirements can be found [.docker/req.txt](.docker/req.txt)

## Python Only Instructions

From project directory:

### Install Requirements

```bash
pip install --no-cache-dir -r .docker/req.txt
```


### Run example

```bash
python3 python/solver.py
```


## Docker Instructions


### Build Project

From the project directory:

```bash
cd .docker
docker compose build
```

### Run Solver Example

```bash
cd .docker
docker compose up solver --remove-orphans --abort-on-container-exit
```

### Run Solver GUI


```bash
cd .docker
docker compose up server --remove-orphans --abort-on-container-exit
```

Then in your browser enter:
```
http://0.0.0.0:5000/
```


### Build Latex Files

```bash
cd .docker
docker compose up latex --remove-orphans --abort-on-container-exit
```
