import os
import psutil
import subprocess

def convert_memory_to_bytes(memory_str):
    """ Convert memory string (e.g., '4G', '1024M') to bytes. """
    if memory_str[-1] in ['G', 'M', 'K', 'B']:
        number = float(memory_str[:-1])
        unit = memory_str[-1]
        if unit == 'G':
            return int(number * 1024 * 1024 * 1024)
        elif unit == 'M':
            return int(number * 1024 * 1024)
        elif unit == 'K':
            return int(number * 1024)
        elif unit == 'B':
            return int(number)
    else:
        return int(memory_str)  # Assume bytes if no unit
    
    
def get_total_memory():
    """ Get memory allocated to the job, considering different schedulers. """
    # Check for Slurm
    if 'SLURM_JOB_ID' in os.environ:
        print("Detecting usage of SLURM")
        try:
            job_id = os.getenv('SLURM_JOB_ID')
            result = subprocess.run(['scontrol', 'show', 'job', job_id], capture_output=True, text=True)
            job_info = result.stdout
            for line in job_info.splitlines():
                if 'MinMemoryNode' in line:
                    mem_allocated = line.split('=')[1]
                    return convert_memory_to_bytes(mem_allocated)
        except Exception as e:
            print(f"Error fetching Slurm memory allocation: {e}")

    # Check for PBS/Torque
    elif 'PBS_JOBID' in os.environ:
        print("Detecting usage of PBS/Torque")
        try:
            job_id = os.getenv('PBS_JOBID')
            result = subprocess.run(['qstat', '-f', job_id], capture_output=True, text=True)
            job_info = result.stdout
            for line in job_info.splitlines():
                if 'Resource_List.mem' in line:
                    mem_allocated = line.split('=')[1].strip()
                    return convert_memory_to_bytes(mem_allocated)
        except Exception as e:
            print(f"Error fetching PBS memory allocation: {e}")

    # Check for SGE
    elif 'SGE_JOB_ID' in os.environ:
        print("Detecting usage of SGE")
        try:
            job_id = os.getenv('SGE_JOB_ID')
            result = subprocess.run(['qstat', '-j', job_id], capture_output=True, text=True)
            job_info = result.stdout
            for line in job_info.splitlines():
                if 'mem' in line:  # Placeholder; needs proper parsing
                    mem_allocated = line.split(':')[1].strip()
                    return convert_memory_to_bytes(mem_allocated)
        except Exception as e:
            print(f"Error fetching SGE memory allocation: {e}")
        
    # Assume that this is running on a local machine    
    else:
        print("Scheduler-specific memory allocation information not found.")
        print("Checking system memory instead...")
        return psutil.virtual_memory().total
    

def get_used_memory():
    """ Get the current process memory usage. """
    if ('SLURM_JOB_ID' in os.environ) or ('PBS_JOBID' in os.environ) or ('SGE_JOB_ID' in os.environ):
        return psutil.Process(os.getpid()).memory_info().rss
    else:
        return psutil.virtual_memory().used