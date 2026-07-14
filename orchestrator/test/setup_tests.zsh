#!/bin/zsh

# call this script to set up the tests to run
# first input argument passed is the path where the tests should run. If the
# directory does not exist, it will be created
# The second input arguement is which types of tests to prepare
# example: zsh setup_tests.zsh /usr/workspace/$USER/test_path all
# note: use -i.bak format for system portability (linux + macOS)

# modify if the dataset gets reset, don't need to change in every input file
# run orchestrator/test/shared_inputs/add_sample_configs_to_storage.py to
# generate these datasets
TEST_Si_DATASET_HANDLE="DS_XXXXXXXXXXXX_0"
TEST_Ta_DATASET_HANDLE="DS_XXXXXXXXXXXX_0"
STORAGE_CREDENTIAL_PATH="/PATH/TO/unittests_colabfit_credentials.json"

ASYNCH_SCHEDULER="SLURM"
ASYNCH_SCHEDULER_HYBRID="FLUX"
CHIMES_LSQ="/PATH/TO/chimes_lsq/build/chimes_lsq"
CHIMES_LSQ_PY="/PATH/TO/chimes_lsq/src/chimes_lsq.py"
HPC_ACCOUNT='ACCOUNT_STRING'
HPC_QUEUE='QUEUE_NAME'
KIM_API='kim-api-collections-management'
LAMMPS_PATH='/PATH/TO/lmp'
NODES=1
TASKS=112
NODES_HYBRID=1
TASKS_HYBRID=1
QE_PATH='/PATH/TO/pw.x'
USE_GPU="false"
FLUX_MACHINE_NAME='NAME_OF_FLUX_MACHINE'

# read paths
INSTALL_PATH=${1}
# all, descriptor, oracle, potential, simulator, storage, target_property
TESTS=${2}
TEST_PATH=$(pwd)

if [[ ${#} -ne 2 || ${INSTALL_PATH} == "-h" ]]; then
    echo "Usage: setup_tests.zsh [path to install tests] [test type]"
    echo "    test type should be one of:"
    echo "      all, descriptor, oracle, potential, score, simulator, storage, target_property"
    exit
fi

if [[ ! "${TESTS}" =~ ^(all|descriptor|oracle|potential|score|simulator|storage|target_property)$ ]]; then
    echo "The third argument must be one of:"
    echo "    all, descriptor, oracle, potential, simulator, storage, target_property"
    exit
fi

# make sure install path exists
if [[ ! -d ${INSTALL_PATH} ]]; then
    mkdir -p ${INSTALL_PATH}
fi

# convert to absolute paths
INSTALL_PATH=$(echo $(realpath ${INSTALL_PATH}))
TEST_PATH=$(realpath ${TEST_PATH})

copy_sample_configs() {
    if [[ ! -d ${INSTALL_PATH}/sample_configs ]]; then
        cp -r shared_inputs/sample_configs ${INSTALL_PATH}
    fi
}

copy_multiel_sample_configs() {
    if [[ ! -d ${INSTALL_PATH}/multi_element_configs ]]; then
        cp -r shared_inputs/multi_element_configs ${INSTALL_PATH}
    fi
}

copy_sample_traj() {
    if [[ ! -d ${INSTALL_PATH}/sample_npt_traj ]]; then
        cp -r shared_inputs/sample_npt_traj ${INSTALL_PATH}
    fi
}

copy_dir_content() {
    MODULE=${1}
    COPY_DEST=${INSTALL_PATH}/${MODULE}/
    mkdir -p ${COPY_DEST}
    cd ${MODULE}
    if [[ -d test_inputs ]]; then
        cp -r test_inputs ${COPY_DEST}
    fi
    if [[ -f driver.py ]]; then
        # trainer seds a driver file
        cp driver.py ${COPY_DEST}
    fi
    if [[ ${MODULE} != 'storage' ]]; then
        # storage doesn't have a test_ file
        cp test_*.py ${COPY_DEST}
    fi
    if [[ -d templates ]]; then
        # only some test modules use templates folders
        cp -r templates ${COPY_DEST}
    fi
    cd ../
}

substitute_inputs() {
    find ./ -name "*.json" | xargs sed -i.bak \
        -e "s|<ASYNCH_SCHEDULER>|${ASYNCH_SCHEDULER}|g" \
        -e "s|<CHIMES_LSQ>|${CHIMES_LSQ}|g" \
        -e "s|<CHIMES_LSQ_PY>|${CHIMES_LSQ_PY}|g" \
        -e "s|<FLUX_MACHINE_NAME>|${FLUX_MACHINE_NAME}|g" \
        -e "s|<HPC_ACCOUNT>|${HPC_ACCOUNT}|g" \
        -e "s|<HPC_QUEUE>|${HPC_QUEUE}|g" \
        -e "s|<INSTALL_PATH>|${INSTALL_PATH}|g" \
        -e "s|<KIM_API>|${KIM_API}|g" \
        -e "s|<LAMMPS_PATH>|${LAMMPS_PATH}|g" \
        -e "s|<NODES>|${NODES}|g" \
        -e "s|<TASKS>|${TASKS}|g" \
        -e "s|<NODES_HYBRID>|${NODES_HYBRID}|g" \
        -e "s|<POTENTIAL_DIR>|${TEST_PATH}/shared_inputs/potential/|g" \
        -e "s|<QE_PATH>|${QE_PATH}|g" \
        -e "s|<STORAGE_CREDENTIAL_PATH>|${STORAGE_CREDENTIAL_PATH}|g" \
        -e "s|<TEST_Si_DATASET_HANDLE>|${TEST_Si_DATASET_HANDLE}|g" \
        -e "s|<TEST_Ta_DATASET_HANDLE>|${TEST_Ta_DATASET_HANDLE}|g" \
        -e "s|<USE_GPU>|${USE_GPU}|g"
	rm *.bak
}

# descriptor tests
if [[ ${TESTS} == "descriptor" || ${TESTS} == "all" ]]; then
    copy_sample_configs
    copy_multiel_sample_configs
    copy_dir_content descriptor

    # specify machine specific params in input files
    cd ${INSTALL_PATH}/descriptor/test_inputs
    substitute_inputs
    # return to starting point
    cd ${TEST_PATH}
fi

# oracle tests
if [[ ${TESTS} == "oracle" || ${TESTS} == "all" ]]; then
    copy_sample_configs
    copy_dir_content oracle
    # specify machine specific params in input files
    cd ${INSTALL_PATH}/oracle/test_inputs
    substitute_inputs
    # set pseduo path for QE setup
    sed -i.bak "s;PSEUDO_DIR;${TEST_PATH}/shared_inputs/pseudos/;" \
        ../templates/espresso.in && rm ../templates/espresso.in.bak
    # return to starting point
    cd ${TEST_PATH}
fi

# score tests
if [[ ${TESTS} == "score" || ${TESTS} == "all" ]]; then
    copy_sample_configs
    copy_dir_content score
    # no machine specific params in input files to change

	# additional settings for FIM tests
	# install potential for testing
	TEST_MODEL_DRIVER='SW__MD_335816936951_005'
	TEST_MODEL='SW_StillingerWeber_1985_Si__MO_405512056662_006'
	# extract from potential file in test_inputs
	tar -xf score/test_inputs/$TEST_MODEL_DRIVER.txz
	tar -xf score/test_inputs/$TEST_MODEL.txz
	# install potential
	${KIM_API} install user ${TEST_MODEL_DRIVER}/
	${KIM_API} install user ${TEST_MODEL}/
	# remove the extracted potential file, since we don't need it anymore
	rm -r ${TEST_MODEL_DRIVER}
	rm -r ${TEST_MODEL}
	# substitute paths
	cd ${INSTALL_PATH}/score/test_inputs
	substitute_inputs
    # return to starting point
    cd ${TEST_PATH}
fi

# simulator tests
if [[ ${TESTS} == "simulator" || ${TESTS} == "all" ]]; then
    copy_sample_configs
    copy_dir_content simulator
    # specify machine specific params in input files
    cd ${INSTALL_PATH}/simulator/test_inputs
    substitute_inputs
    # return to starting point
    cd ${TEST_PATH}
fi

# storage tests
if [[ ${TESTS} == "storage" || ${TESTS} == "all" ]]; then
    copy_sample_configs
    copy_dir_content storage

    cd ${INSTALL_PATH}/storage/test_inputs
    substitute_inputs
    # return to starting point
    cd ${TEST_PATH}
fi

# skip target property for now (wait Fitsnap merge)
if [[ ${TESTS} == "target_property" || ${TESTS} == "all" ]]; then
    copy_sample_configs
    copy_sample_traj
    copy_dir_content target_property
    cd ${INSTALL_PATH}/target_property/test_inputs

	substitute_inputs
    # update the forcefield directory for target property simulations with the full path
    cd ../templates
    sed -i.bak "s;FORCEFIELD_DIR;${TEST_PATH}/shared_inputs/forcefield;" \
        npt.in
    sed -i.bak "s;FORCEFIELD_DIR;${TEST_PATH}/shared_inputs/forcefield;" \
        nph.in
    rm *.bak

    # return to starting point
    cd ${TEST_PATH}
fi

# trainer (and potential) tests
if [[ ${TESTS} == "potential" || ${TESTS} == "all" ]]; then
    copy_dir_content potential
    # specify machine specific params in input files
    cd ${INSTALL_PATH}/potential/test_inputs
    substitute_inputs
    # return to starting point
    cd ${TEST_PATH}
fi

# set the reference and test paths in the pytest files
cd ${INSTALL_PATH}
if [[ ${MODULE} != 'storage' ]]; then
    # storage doesn't have any of these files
    find ./ -name "test_*.py" | xargs sed -i.bak "s|INSTALL_PATH|${INSTALL_PATH}|"
    find ./ -name "test_*.py" | xargs sed -i.bak "s|TEST_PATH|${TEST_PATH}|"
    find ./ -name "*.bak" | xargs rm
fi

cd ${INSTALL_PATH}
cp ${TEST_PATH}/run_all.zsh .
chmod u+x run_all.zsh
