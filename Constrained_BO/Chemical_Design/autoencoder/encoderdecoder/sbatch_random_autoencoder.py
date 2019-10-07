import shutil
import subprocess
import time
import os

# from http://stackoverflow.com/questions/4814970/subprocess-check-output-doesnt-seem-to-exist-python-2-6-5
if "check_output" not in dir(subprocess):  # duck punch it in!
    def f(*popenargs, **kwargs):
        if 'stdout' in kwargs:
            raise ValueError('stdout argument not allowed, it will be overridden.')
        process = subprocess.Popen(stdout=subprocess.PIPE, *popenargs, **kwargs)
        output, unused_err = process.communicate()
        retcode = process.poll()
        if retcode:
            cmd = kwargs.get("args")
            if cmd is None:
                cmd = popenargs[0]
            raise subprocess.CalledProcessError(retcode, cmd)
        return output
    subprocess.check_output = f


NJOBS = 200
job_filename = 'ml_job.sh'
total_errors = 0

for i in range(NJOBS):
    origin = '/n/home05/rgbombarelli/random_autoencoder_experiment'
    destination = origin + '_' + str(i) + '_' + time.strftime('%y-%m-%d-%H-%M-%S')
    shutil.copytree(origin, destination)
    os.chdir(destination)
    name = destination.split('/')[-1]
    cmd = "sbatch --job-name={name} {jobfile}".format(name=name, jobfile=job_filename)
    try:
        out = subprocess.check_output(cmd, shell=True)
    except subprocess.CalledProcessError, e:
        print "Problem (count={0}) calling sbatch: {1}".format(total_errors, e)
        if total_errors > 100:
            raise
    time.sleep(1)
