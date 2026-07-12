Factory / Builder
=================

In order to facilitate the modular and interchangeable design approach of the Orchestrator modules, we provide class factories and builders for instantiating concrete classes in a uniform manner. There is a single generalized :class:`~orchestrator.utils.module_factory.ModuleFactory` class which is invoked and specified in each independent module as::

   oracle_factory = ModuleFactory(Oracle)
   oracle_factory.add_new_module('QE', EspressoOracle)

Note that the factory function arguments are classes themselves, not instances
of classes. While a single module factory could be used across the board, we
define specific factories based on the abstract base classes of each module, to
avoid possible naming conflicts (such as a LAMMPS oracle and LAMMPS simulator).
The available concrete modules are added to the factory using the :meth:`~orchestrator.utils.module_factory.ModuleFactory.add_new_module` method. Classes can then be instantiated using module-specific builders, based off of the abstract base
class :class:`~orchestrator.utils.module_factory.ModuleBuilder`. Users and
other modules will typically only need to interface with a builder, unless they
are creating a new concrete class that they would like to add to the factory.
At the end of every ``factory.py`` file defined by each module, a standard
builder is created which can be imported and used across the Orchestrator::

   oracle_builder = OracleBuilder()
   ...
   from orchestrator.oracle import oracle_builder

The builder and factory classes provide a common interface from which specific
and concrete implementations of the modules can be instantiated, where the only
change of code needed to switch to a different implementation is passing a
different token to the builder::

   instantiated_oracle = oracle_builder('QE', oracle_creation_arguments)

See the full API for the module at :ref:`factory_module`.

Inheritance Graph
-----------------

.. inheritance-diagram::
   orchestrator.utils.module_factory
   orchestrator.computer.descriptor.factory
   orchestrator.computer.score.factory
   orchestrator.oracle.factory
   orchestrator.potential.factory
   orchestrator.scheduler.factory
   orchestrator.simulator.factory
   orchestrator.storage.factory
   orchestrator.target_property.factory
   orchestrator.trainer.factory
   :parts: 3
