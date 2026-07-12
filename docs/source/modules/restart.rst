Restart
=======

Considering that usage of the Orchestrator may include long-walltime jobs or multistep pipelines, it is useful to have the ability to restart the Orchestrator in some specific state beyond basic initialization. To this end, we have created restart functionality that is flexible and extensible, defined on a per-module basis. Generally speaking, one simply needs to import the
:class:`~orchestrator.utils.restart.restarter` instance to leverage the
:meth:`~orchestrator.utils.restart.Restart.write_checkpoint_file` and
:meth:`~orchestrator.utils.restart.Restart.read_checkpoint_file` methods which
are central to the restart methodology.

The primary purpose of the Restart module is to save the current state of the
Orchestrator to enable discontinuous operation. As with the overall design of
the Orchestrator itself, the restart functionality is intended to be modular
and decentralized - each module is responsible to defining and handling its own
checkpointing and restart behavior. Some modules may not require such
capabilities at all. By convention, a module which includes restart
capabilities needs to define two methods: ``checkpoint_[module name]()`` and
``restart_[module name]()``.

See the full API for the module at :ref:`restart_module`.

.. _restart_structure:

Checkpoint File Structure
-------------------------

While modules are free to define their own auxiliary files to assist in the
restart process (see Schedulers
:meth:`~.scheduler_base.Scheduler.save_job_dict` as an
example), the bulk of the information will be saved in a shared json file. This
file is organized in a hierarchical fashion - at the highest level it is split
up into sections which correspond to each module instance. Each of these
sections can also be organized, but this level of oragnization is imposed by
the individual modules. Generically, this leads to a file structured as:

.. code-block:: none

    "module_name_1":{
        "item1": data,
        "item2": data,
        ...
    },
    "module_name_2":{
        "item1": data,
        "hierarchical item2":{
            "item1": data,
            "item2": data
        },
        ...
    },
    ...

While the file should generally be left alone (since Orchestrator will automatically read and write from it as necessary), it is still human readable. This allows for one to inspect the current state of the Orchestrator, and advanced users may find it convenient to modify this file directly to induce a desired state upon restart.

Naming Conventions
------------------

By default, modules using restart functionality define a file name where the
checkpointed information is written: ``./orchestrator_checkpoint.json``. This
file will be shared by all modules and its structure is discussed in greater
detail in :ref:`restart_structure`. This filename can be overridden by adding
the ``checkpoint_file`` field in the input arguments for constructing the
module::

    module = module_builder.build(
        'TOKEN',
        {
            'module_arg1': value1,
            'checkpoint_file': 'custom_name.json',
        },
    )

In addition to the checkpoint file name itself, modules also define the name
they will use to demarcate their section of the checkpoint file, with the
default value as the module type. For instance any :class:`~.Potential` module
uses a default of "potential". This value can be changed by specifying a
``checkpoint_name`` in the input arguments for that module::

    module = module_builder.build(
        'TOKEN',
        {
            'module_arg1': value1,
            'checkpoint_name': 'custom_section_name',
        },
    )

.. warning::
    If your application uses multiple instances of a given module type, you
    should change at least one of their ``checkpoint_name``\ s, otherwise they
    will overwrite each other's checkpoint information. Due to the common usage
    of multiple different :class:`~.Scheduler` modules for complex Orchestrator
    operations, this module sets the ``checkpoint_name`` to the specific class
    name be default to avoid common collisions. If using multiple instances of
    the same class, the ``checkpoint_name`` should be manually overridden for
    at least one of the instances.

Use Cases
---------

Read
^^^^

Generally, the "restart" machinary should only be used at instantiation/start up: :meth:`~orchestrator.utils.restart.Restart.read_checkpoint_file` should be called as the last step of the module's ``__init__()`` function. In this way, if any information is available in the ``checkpoint_file`` under the proper ``checkpoint_name``, it can properly initialize or update variables based on the last checkpoint.

Write
^^^^^

More discretion can be used regarding when a module should write to the
``checkpoint_file``. In the case of the
:class:`~.scheduler_base.Scheduler` classes, the checkpoint
file is updated any time a JobStatus is updated. On the other hand,
the :class:`~orchestrator.potential.potential_base.Potential` module never
calls its own
:meth:`~orchestrator.potential.potential_base.Potential.checkpoint_potential`
method, but relies on other functions which handle logic around the potential
to inform when the checkpoint should be written.

For some modules, checkpointing acts as a simple way to save the "memory" of
the module, while for others, checkpointing enables the discontinuous execution
of more complex or time-intensive operations. In these cases, logic must be
integrated into methods which change their behavior based on the state of flags
designed to track the progress. See :meth:`~.MeltingPoint.calculate_property`
for an example.

Inheritance Graph
-----------------

.. inheritance-diagram::
   orchestrator.utils.restart
   :parts: 3
