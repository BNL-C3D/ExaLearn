import os
import sys
import time
import pprint

import radical.pilot as rp
import radical.utils as ru

dh = ru.DebugHelper()

os.environ['RADICAL_VERBOSE'] = 'DEBUG'
#os.environ['RADICAL_VERBOSE'] = 'REPORT'
    
report = ru.Reporter(name='radical.pilot')
report.title('Getting Started (RP version %s)' % rp.version)

PWD = os.path.abspath(os.path.dirname(__file__))
RESOURCE = 'xsede.stampede2_ssh'
PROJECT  = 'TG-MCB090174'

# get a pre-installed resource configuration
session = rp.Session()
cfg = session.get_resource_config(RESOURCE)
pprint.pprint (cfg)

try:

    # Add a Pilot Manager. Pilot managers manage one or more ComputePilots.
    pmgr = rp.PilotManager(session=session)
    pd_init = {'resource'      : RESOURCE,
               'runtime'       : 60,  # pilot runtime (min)
               'exit_on_error' : True,
               'project'       : PROJECT,
               'queue'         : cfg['default_queue'],
               'access_schema' : cfg['schemas'][0], 
               'cores'         : 4,
              }
    pdesc = rp.ComputePilotDescription(pd_init)
    
    # Launch the pilot.
    pilot = pmgr.submit_pilots(pdesc)

    report.header('submit units')
    umgr = rp.UnitManager(session=session)
    umgr.add_pilots(pilot)

    n = 5 ## number of units to run
    p_num = 4 ## number of processes (MPI)
    report.info('create %d unit description(s)\n\t' % n)

    cuds = list()
    for i in range(0, n):
        # create a new CU description, and fill it.
        # Here we don't use dict initialization.
        cud = rp.ComputeUnitDescription()
        cud.executable       = '/home1/06078/tg853774/er_graph/er'
        cud.arguments        = [1000, 100, 0.05]
        #cud.input_staging    = ['%s/09_mpi_units.sh' % PWD]
        #cud.executable       = 'python'
        #cud.arguments        = ['helloworld_mpi.py']
        #cud.input_staging    = ['%s/helloworld_mpi.py' % PWD]
        cud.gpu_processes    = 0
        cud.cpu_processes    = 3
        cud.cpu_process_type = 'mpi'
#        cud.cpu_threads      = 1
#        cud.cpu_process_type = rp.POSIX
#        cud.cpu_thread_type  = rp.POSIX
        cuds.append(cud)
        report.progress()
    report.ok('>>ok\n')

    # Submit the previously created ComputeUnit descriptions to the
    # PilotManager. This will trigger the selected scheduler to start
    # assigning ComputeUnits to the ComputePilots.
    units = umgr.submit_units(cuds)

    # Wait for all compute units to reach a final state (DONE, CANCELED or FAILED).
    report.header('gather results')
    umgr.wait_units()

    report.info('\n')
    for unit in units:
        report.plain('  * %s: %s, exit: %3s, ranks: %s\n'
                % (unit.uid, unit.state[:4], unit.exit_code, unit.stdout))
        # ranks = list()
        # for line in unit.stdout.split('\n'):
        #     if line.strip():
        #         rank = line.split()[1]
        #         ranks.append(rank)
        # for p in range(p_num):
        #     for t in range(t_num):
        #         rank = '%d:%d/1' % (p, t)
        #         assert(rank in ranks), 'missing rank %s' % rank

except Exception as e:
    # Something unexpected happened in the pilot code above
    report.error('caught Exception: %s\n' % e)
    ru.print_exception_trace()
    raise

except (KeyboardInterrupt, SystemExit) as e:
    # the callback called sys.exit(), and we can here catch the
    # corresponding KeyboardInterrupt exception for shutdown.  We also catch
    # SystemExit (which gets raised if the main threads exits for some other
    # reason).
    ru.print_exception_trace()
    report.warn('exit requested\n')

finally:
    # always clean up the session, no matter if we caught an exception or
    # not.  This will kill all remaining pilots.
    report.header('finalize')
    session.close(download=True)

report.header()

