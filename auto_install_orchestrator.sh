#!/usr/bin/env bash
##############################################################################
# Robust installer for the “orchestrator” micromamba environment
# To install with your current git branch, call from 1 directory up
# i.e. ENV_NAME=[name] orchestrator/auto_install_orchestrator.sh
# alternatively, you can pass REPO_DIR to your git top directory
# For a default install using the develop branch, no arguments are needed if
# ran in a blank directory for the environment and repo clone to be placed.
# Will take ~10 GB of space.
# – use source_me.sh to re-enter the environment
##############################################################################
set -Eeuo pipefail
trap 'echo "[ERROR] ${BASH_SOURCE[0]}:${LINENO} – aborting"; exit 1' ERR

# ---------- CONFIG ----------------------------------------------------------
ENV_NAME=${ENV_NAME:-orchestrator} # override with ENV_NAME=myenv ./install.sh
PY_VER=3.10
ROOT_PREFIX="${PWD}"               # micromamba root lives beside this script
MICROMAMBA_BIN="${PWD}/bin/micromamba"
REPO_URL="https://github.com/LLNL/orchestrator.git"
REPO_DIR="orchestrator" # where the repo will be cloned
KIM_API_ENVIRONMENT_COLLECTION="/PATH/TO/SHARED/ENVIRONMENT/COLLECTION/DIR" # for accessing shared potentials
STARTUP_FILE="source_me.sh"
# note: if lammps is not yet installed, leave these blank
#   or specify the paths where the installation will be accessible
#   if blank, they will need to be manually adjusted in the STARTUP_FILE later
#   if compiling lammps, be sure to include cython (`pip install cython`) in your env
LAMMPS_PYTHON_MODULE="/PATH/TO/LAMMPS/python" # python directory of the lammps code
LAMMPS_LIBRARY_PATH="/PATH/TO/LAMMPS/BUILD/" # where the compiled lammpslib.so sits

# test if mpicc can be found
if which mpicc &>/dev/null; then
    MPI_LIB="$(dirname $(dirname $(which mpicc)))/lib"
else
    # if mpicc not available in mpi library, can hard code path here
    MPI_LIB="<LIB for MPI>"
fi

# ----------------------------------------------------------------------------

echo "[INFO] Starting installer at $(date)"

# known bug where micromamba reads from .condarc, temporarily move it
echo "[INFO] temporarily moving .condarc file..."
echo "[INFO] (if this script does not complete, manually run 'mv ~/.condarcbackup ~/.condarc')"
mv ~/.condarc ~/.condarcbackup

# ---------- prerequisite sanity checks -------------------------------------
for cmd in curl tar git; do
  command -v "$cmd" >/dev/null || { echo "[FATAL] <<$cmd>> not found"; exit 1; }
done

# ---------- download micromamba if missing ----------------------------------
if [[ ! -x "${MICROMAMBA_BIN}" ]]; then
  echo "[INFO] Downloading micromamba -> ${MICROMAMBA_BIN}"
  mkdir -p "$(dirname "${MICROMAMBA_BIN}")"

  # determine computer type
  case "$(uname)" in
    Linux)
      PLATFORM="linux" ;;
    Darwin)
      PLATFORM="osx" ;;
    *NT*)
      PLATFORM="win" ;;
  esac

  ARCH="$(uname -m)"
  case "$ARCH" in
    aarch64|ppc64le|arm64)
        ;;  # pass
    *)
      ARCH="64" ;;
  esac

  case "$PLATFORM-$ARCH" in
    linux-aarch64|linux-ppc64le|linux-64|osx-arm64|osx-64|win-64)
        ;;  # pass
    *)
      echo "Failed to detect your OS" >&2
      exit 1
      ;;
  esac

  curl -Ls https://micro.mamba.pm/api/micromamba/${PLATFORM}-${ARCH}/latest | tar -xvj bin/micromamba
fi

# shell hook (posix syntax works for bash/zsh/sh)
export MAMBA_ROOT_PREFIX="${ROOT_PREFIX}"
eval "$("${MICROMAMBA_BIN}" shell hook -s posix)"

# ---------- create env if absent -------------------------------------------
if "${MICROMAMBA_BIN}" env list | grep -qE "^[* ] ${ENV_NAME}\s"; then
  echo "[WARN] Env <<${ENV_NAME}>> already exists"
  # give option to make a new name
  read -p "Would you like to use a different name? (yes/no): " RESPONSE
  if [[ "${RESPONSE}" == "yes" ]]; then
    read -p "Please enter the alternate environment name: " NEW_ENV_NAME
    if [[ -n "${NEW_ENV_NAME}" ]]; then
      ENV_NAME="${NEW_ENV_NAME}"
      # make sure this is actually new
      if "${MICROMAMBA_BIN}" env list | grep -qE "^[* ] ${ENV_NAME}\s"; then
        echo "The environment name '${ENV_NAME}' also exists. Please choose a unique name."
        exit 1
      fi
    else
      echo "No alternate name provided. Exiting..."
      exit 1
    fi
  else
    echo "Please remove the environment before continuing"
    exit 1
  fi
fi
# ENV_NAME is safe
echo "[INFO] Creating env <<${ENV_NAME}>> (python=${PY_VER})"
"${MICROMAMBA_BIN}" create -y -n "${ENV_NAME}" \
    "python=${PY_VER}" "cmake<4" -c conda-forge

# ---------- activate env ----------------------------------------------------
echo "[INFO] Activating ${ENV_NAME}"
set +u
micromamba activate "${ENV_NAME}"

# ---------- package install -------------------------------------------------

echo "[INFO] Installing conda-forge deps"
micromamba install -y \
  ase=3.25.0 \
  h5py=3.13.0 \
  kimpy=2.1.3 \
  kim-api=2.4.1 \
  kim-edn=1.4.1 \
  pandas=2.2.3 \
  pytest=8.3.5 \
  scikit-learn=1.6.1 \
  scipy=1.15.2 \
  pre-commit sphinx furo \
  -c conda-forge

# torch, colabfit, fitsnap and optional dependencies will be installed with pip
# during the orchestrator install
# NOTE: lammps should NOT be installed through conda or pip

echo "[INFO] Initial dependencies completed at $(date)"

# ---------- KIM-API ---------------------------------------------------------

# this should be the LD path for kim-api
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:${MAMBA_ROOT_PREFIX}/envs/${ENV_NAME}/lib/"
if [[ -d ${KIM_API_ENVIRONMENT_COLLECTION} ]]; then
  export KIM_API_MODEL_DRIVERS_DIR="${KIM_API_ENVIRONMENT_COLLECTION}/model-drivers"
  export KIM_API_PORTABLE_MODELS_DIR="${KIM_API_ENVIRONMENT_COLLECTION}/portable-models"
  export KIM_API_SIMULATOR_MODELS_DIR="${KIM_API_ENVIRONMENT_COLLECTION}/simulator-models"
fi

# ---------- orchestrator repo ----------------------------------------------
if [[ -d "${REPO_DIR}/.git" ]]; then
  echo "[INFO] Updating <<${REPO_DIR}>>"
  git -C "${REPO_DIR}" pull --ff-only
else
  echo "[INFO] Cloning orchestrator"
  git clone "${REPO_URL}" "${REPO_DIR}"
fi

echo "[INFO] Installing orchestrator (editable) + extras - this step may take ~10 min"
# install all optional dependencies
pip install --quiet --no-cache-dir -e "${REPO_DIR}[AIIDA, LTAU, FIMMATCHING, NEQUIP]"

echo "[INFO] Orchestrator installation completed at $(date)"

# ---------- generate source_me.sh   --------------------------------------
cat > "${STARTUP_FILE}" <<EOS
# Source this after login to activate the orchestrator environment
# Usage:  source ./source_me.sh

# --- locate this project ---------------------------------------------------
SCRIPT_DIR="\$( cd -- "\$( dirname -- "\${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export PATH="\${SCRIPT_DIR}/bin:\${PATH}"
export MAMBA_ROOT_PREFIX="\${SCRIPT_DIR}"

# --- activate the environment ----------------------------------------------
eval "\$("\${SCRIPT_DIR}/bin/micromamba" shell hook -s bash)"
micromamba activate ${ENV_NAME}

# --- set environment variables ---------------------------------------------
# set LD_LIBRARY_PATH for MPI (mpi4py need)
export LD_LIBRARY_PATH="${MPI_LIB}:\${LD_LIBRARY_PATH}"
# set LD_LIBRARY_PATH for the KIM-API
export LD_LIBRARY_PATH="${MAMBA_ROOT_PREFIX}/envs/${ENV_NAME}/lib/:\${LD_LIBRARY_PATH}"
# set LD_LIBRARY_PATH for LAMMPS
export LD_LIBRARY_PATH="${LAMMPS_LIBRARY_PATH:-PATH_TO_liblammps.so}:\${LD_LIBRARY_PATH}"
# set PYTHONPATH for LAMMPS
export PYTHONPATH="\${PYTHONPATH:-}:${LAMMPS_PYTHON_MODULE:-PATH_TO_LAMMPS/PYTHON}"

if [[ -d ${KIM_API_ENVIRONMENT_COLLECTION} ]]; then
  export KIM_API_MODEL_DRIVERS_DIR="${KIM_API_ENVIRONMENT_COLLECTION}/model-drivers"
  export KIM_API_PORTABLE_MODELS_DIR="${KIM_API_ENVIRONMENT_COLLECTION}/portable-models"
  export KIM_API_SIMULATOR_MODELS_DIR="${KIM_API_ENVIRONMENT_COLLECTION}/simulator-models"
fi
# set CMAKE_PREFIX_PATH for KIM-API
export CMAKE_PREFIX_PATH="${MAMBA_ROOT_PREFIX}/envs/${ENV_NAME}/share/cmake"
EOS
chmod +x "${STARTUP_FILE}"
echo "[INFO] Startup helper written → $(realpath "${STARTUP_FILE}")"

# ---------- initialize pre-commit ------------------------------------------
echo "[INFO] Initialising pre-commit hooks"
pushd "${REPO_DIR}" >/dev/null
pre-commit install --install-hooks
popd >/dev/null

# ---------- restore .condarc file ------------------------------------------
echo "[INFO] resetting .condarc file to original file"
mv ~/.condarcbackup ~/.condarc

# ---------- kimkit user setup ----------------------------------------------
echo "[INFO] Attempting to initialize user in kimkit repo. If not configured, this will timeout"
USER=`whoami`
python - <<PY
try:
    from kimkit import users
    try:
        users.add_self_as_user("${USER}")
    except RuntimeError as e:
        print(e)
except Exception as e:
    print(e)
PY

echo "[INFO] Completed installation at $(date)"
echo -e "\n[SUCCESS] Done!  In a new terminal run:  source $(pwd)/${STARTUP_FILE}\n"
