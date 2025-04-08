# multirobot_inspection_optimizer

This project showcases a MILP solver for a ground and aerial robot fleet inspection route planning problem.

## Requirements

Requires docker and docker compose to run.

## Build

From the project directory:

```bash
cd .docker
docker compose build
```


## Run

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