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
            node_interface=self.node_interface,
            scheduler_options={'interface': self.scheduler_interface,
                               'port': self.port},
            log_directory=os.getcwd()
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
        if self.cluster:
            self.cluster.close()
            self.cluster = None

    def __del__(self):
        self.close()


cluster = SLURM()
