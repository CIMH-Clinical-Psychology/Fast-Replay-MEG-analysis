#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 16 11:17:04 2026

@author: simon.kern
"""

import os
import atexit
from dask.distributed import Client
from dask_jobqueue import SLURMCluster

class Local:
    """a pass-through cluster that only runs locally with n_jobs"""

class SLURM:
    """
    Manages Dask SLURMCluster and Client lifecycle.

    Parameters
    ----------
    queue : str
        SLURM queue name.
    cores : int
        Number of cores per job.
    memory : str
        Memory limit per job.
    walltime : str
        Job time limit.
    interface : str
        Network interface for workers.
    scheduler_interface : str
        Network interface for the scheduler.
    port : int
        Scheduler port.
    jobs : int
        Number of jobs to scale.
    """
    def __init__(self, queue='cpu', cores=4, memory='16GB',
                 walltime='01:00:00', node_interface='eno1',
                 scheduler_interface='ens3', port=60500, jobs=1):
        self.queue = queue
        self.cores = cores
        self.memory = memory
        self.walltime = walltime
        self.node_interface = node_interface
        self.scheduler_interface = scheduler_interface
        self.port = port
        self.jobs = jobs
        self.cluster = None
        self.client = None
        # Ensure cleanup on interpreter exit
        atexit.register(self.close)

    def __enter__(self):
        self.cluster = SLURMCluster(
            queue=self.queue,
            cores=self.cores,
            memory=self.memory,
            walltime=self.walltime,
            interface=self.node_interface,
            scheduler_options={'interface': self.scheduler_interface,
                               'port': self.port},
            log_directory='/zi/home/simon.kern/'
        )
        self.cluster.scale(jobs=self.jobs)
        self.client = Client(self.cluster)

        self.client.wait_for_workers(n_workers=self.jobs)
        return self.client

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        """Shut down client and cluster resources."""
        if self.client:
            self.client.close()
            self.client = None
        else:
            print('no open client')
        if self.cluster:
            self.cluster.close()
            self.cluster = None
        else:
            print('no open cluster')

    def __del__(self):
        self.close()


def dummy_task(x):
    """
    Perform a dummy computation.
    """
    import time
    import numpy as np
    time.sleep(1)
    return np.sqrt(x) ** 2

if __name__ == "__main__":
    from joblib import Parallel, delayed, register_parallel_backend
    from joblib import parallel_backend
    # The 'with' block for SLURM ensures the Client is active.
    # Parallel(backend='dask') will hook into that active Client.
    client = SLURM(jobs=2, cores=2)


    with client:
        results = Parallel(backend='dask')(delayed(dummy_task)(i) for i in range(10))

    print(results)
