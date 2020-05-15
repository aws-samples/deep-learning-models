import paramiko
from scp import SCPClient
from concurrent.futures import ThreadPoolExecutor
from time import time
import os

class SSH(object):
    """
    Provides a simple set of ssh and scp tools for working
    with one or several ec2 instances.
    """
    def __init__(self, public_ips, key_file, user_name='ubuntu', async_threads=None):
        """
        Parameters
        ----------
        public_ips : list(str)
            List of public ip addresses for EC2 instances
        key_file : str
            filepath to keyfile
        user_name : str optional
            user name to login to instances, default is ubuntu
        async_threads : int
            maximum number of threads, default is number of instances * 8
        """
        self.public_ips = public_ips
        self.key = key_file
        self.user_name = user_name
        self.thread_pool = ThreadPoolExecutor(max_workers=async_threads \
            if async_threads \
            else len(self.public_ips) * 8)

    def create_connection(self):
        """
        Creates a generic ssh connection

        Returns
        -------
        ssh : paramiko.SSHClient
        """
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        return ssh

    def run_on_node(self, hostname, command):
        """
        Run ssh command on a given host

        Parameters
        ----------
        hostname : str
            public ip address of node
        command : str
            command to run on node

        Returns
        -------
        results : dict
            a dict of stdout and stderr result from command
        """
        ssh = self.create_connection()
        ssh.connect(hostname=hostname, username=self.user_name,
                         key_filename=self.key)
        stdin, stdout, stderr = ssh.exec_command(command)
        results = {'stdout': ''.join(stdout.readlines()),
                   'stderr': ''.join(stderr.readlines())}
        ssh.close()
        return results

    def run_on_master(self, command, wait=True):
        """
        Run a command on the master node (first ip in list)

        Parameters
        ----------
        command : str
            The command to run
        wait : bool optional default True
            If true, submits thread to scheduler and returns thread
            without waiting for result

        Returns
        -------
        result : dict or thread
        if wait==True returns dict of stdout and stderr
        if wait==False returns thread object without waiting for results

        """
        task = self.thread_pool.submit(self.run_on_node, self.public_ips[0], command)
        if wait:
            while not task.done():
                continue
            return task.result()
        return task

    def run_on_workers(self, command, wait=True):
        """
        Run a command on all worker nodes

        Parameters
        ----------
        command : str
            the command to run
        wait : bool optional default True
            If true, submits thread to scheduler and returns thread
            without waiting for result

        Returns
        -------
        result : list of dict or thread
        if wait==True returns dict of stdout and stderr
        if wait==False returns thread object without waiting for results
        """
        tasks = [self.thread_pool.submit(self.run_on_node, worker, command) \
                 for worker in self.public_ips[1:]]
        if wait:
            while not all([i.done() for i in tasks]):
                continue
            return [i.result() for i in tasks]
        return tasks

    def run_on_all(self, command, wait=True):
        """
        Run a command on all worker nodes

        Parameters
        ----------
        command : str
            the command to run
        wait : bool optional default True
            If true, submits thread to scheduler and returns thread
            without waiting for result

        Returns
        -------
        result : list of dict or thread
        if wait==True returns dict of stdout and stderr
        if wait==False returns thread object without waiting for results
        """
        '''tasks = [self.thread_pool.submit(self.node_bash, node, command) \
                 for node in self.public_ips]'''
        tasks = [self.thread_pool.submit(self.run_on_node, worker, command) \
                 for worker in self.public_ips]
        if wait:
            while not all([i.done() for i in tasks]):
                continue
            return [i.result() for i in tasks]
        return tasks

    def node_scp_put(self, hostname, src, dest, recursive=False):
        """
        Copy a file from local to a node

        Parameters
        ----------
        hostname
        src
        dest
        recursive

        Returns
        -------

        """
        ssh = self.create_connection()
        ssh.connect(hostname=hostname, username=self.user_name,
                         key_filename=self.key)
        scp = SCPClient(ssh.get_transport())
        results = scp.put(src, remote_path=dest, recursive=recursive)
        ssh.close()
        return results

    def node_scp_get(self, hostname, src, dest="", recursive=False):
        """
        Copy a file from a node to local

        Parameters
        ----------
        hostname
        src
        dest
        recursive

        Returns
        -------

        """
        ssh = self.create_connection()
        ssh.connect(hostname=hostname, username=self.user_name,
                         key_filename=self.key)
        scp = SCPClient(ssh.get_transport())
        results = scp.get(src, dest, recursive)
        ssh.close()
        return results

    def scp_local_to_master(self, src, dest, recursive=False, wait=True):
        """
        Copy a file from local to master

        Parameters
        ----------
        src
        dest
        recursive
        wait

        Returns
        -------

        """
        task = self.thread_pool.submit(self.node_scp_put, self.public_ips[0], src, dest, recursive)
        if wait:
            while not task.done():
                continue
            return task.result()
        return task

    def scp_master_to_local(self, src, dest="", recursive=False, wait=True):
        """
        Copy a file from master to local

        Parameters
        ----------
        src
        dest
        recursive
        wait

        Returns
        -------

        """
        task = self.thread_pool.submit(self.node_scp_get, self.public_ips[0], src, dest, recursive)
        if wait:
            while not task.done():
                continue
            return task.result()
        return task

    def scp_local_to_workers(self, src, dest, recursive=False, wait=True):

        """
        copy a file from local to workers

        Parameters
        ----------
        src
        dest
        recursive
        wait

        Returns
        -------

        """
        tasks = [self.thread_pool.submit(self.node_scp_put, i, src, dest, recursive) \
                 for i in self.public_ips[1:]]
        if wait:
            while not all([i.done() for i in tasks]):
                continue
            return [i.result() for i in tasks]
        return tasks

    def scp_workers_to_local(self, src, dest, recursive=False, wait=True):
        """
        copy a file from workers to local

        Parameters
        ----------
        src
        dest
        recursive
        wait

        Returns
        -------

        """
        tasks = [self.thread_pool.submit(self.node_scp_get, i, src, dest + str(j), recursive) \
                 for j, i in enumerate(self.public_ips[1:])]
        if wait:
            while not all([i.done() for i in tasks]):
                continue
            return [i.result() for i in tasks]
        return tasks

    def scp_local_to_all(self, src, dest, recursive=False, wait=True):

        """
        copy a file from local to workers

        Parameters
        ----------
        src
        dest
        recursive
        wait

        Returns
        -------

        """
        tasks = [self.thread_pool.submit(self.node_scp_put, i, src, dest, recursive) \
                 for i in self.public_ips]
        if wait:
            while not all([i.done() for i in tasks]):
                continue
            return [i.result() for i in tasks]
        return tasks

    def scp_all_to_local(self, src, dest, recursive=False, wait=True):
        """
        copy a file from workers to local

        Parameters
        ----------
        src
        dest
        recursive
        wait

        Returns
        -------

        """
        tasks = [self.thread_pool.submit(self.node_scp_get, i, src, dest + str(j), recursive) \
                 for j, i in enumerate(self.public_ips)]
        if wait:
            while not all([i.done() for i in tasks]):
                continue
            return [i.result() for i in tasks]
        return tasks

def get_gpu_counts(sh):
    counts = sh.run_on_all('nvidia-smi --query-gpu=gpu_name --format=csv | wc -l')
    counts = [int(count['stdout'])-1 for count in counts]
    return counts

def create_hostfile(sh, private_ips, outfile='hosts'):
    gpu_counts = get_gpu_counts(sh)
    slots = {ip: count for ip, count in zip(private_ips, gpu_counts)}
    hosts = ''
    for i, j in slots.items():
        hosts += "{0}\tslots={1}\n".format(i, j)
    sh.run_on_all("printf \"{0}\" >> {1}".format(hosts, outfile))
    return

def create_ssh_comm(sh):
    sh.run_on_master('ssh-keygen -t rsa -N "" -f ${HOME}/.ssh/id_rsa')
    sh.run_on_all('printf "Host *\n\tForwardAgent yes\n\tStrictHostKeyChecking no\n" >> ${HOME}/.ssh/config')
    sh.run_on_all('printf "\tUserKnownHostsFile=/dev/null\n" >> ${HOME}/.ssh/config')
    sh.run_on_all('printf "\tLogLevel=ERROR\n\tServerAliveInterval=30\n" >> ${HOME}/.ssh/config')
    sh.run_on_all('printf "\tUser ubuntu\n" >> ${HOME}/.ssh/config')
    private_key = sh.run_on_master("cat $HOME/.ssh/id_rsa")
    public_key = sh.run_on_master("cat $HOME/.ssh/id_rsa.pub")
    sh.run_on_all('printf "{0}" >> $HOME/.ssh/authorized_keys'.format(public_key['stdout']))
    sh.run_on_workers('echo "{0}" >> $HOME/.ssh/id_rsa'.format(private_key['stdout']))
    sh.run_on_all('chmod 600 $HOME/.ssh/id_rsa')
    return

def setup_container_communication(sh):
    sh.run_on_all('mkdir ssh_container')
    sh.run_on_all('cp hosts ssh_container/')
    sh.run_on_master('ssh-keygen -t rsa -N "" -f ${HOME}/ssh_container/id_rsa')
    sh.run_on_all('printf "Host *\n\tForwardAgent yes\n\tStrictHostKeyChecking no\n" >> ${HOME}/ssh_container/config')
    sh.run_on_all('printf "\tUserKnownHostsFile=/dev/null\n" >> ${HOME}/ssh_container/config')
    sh.run_on_all('printf "\tLogLevel=ERROR\n\tServerAliveInterval=30\n" >> ${HOME}/ssh_container/config')
    sh.run_on_all('printf "\tUser ubuntu\n" >> ${HOME}/ssh_container/config')
    private_key = sh.run_on_master("cat $HOME/ssh_container/id_rsa")
    public_key = sh.run_on_master("cat $HOME/ssh_container/id_rsa.pub")
    sh.run_on_all('printf "{0}" >> $HOME/ssh_container/authorized_keys'.format(public_key['stdout']))
    sh.run_on_workers('echo "{0}" >> $HOME/ssh_container/id_rsa'.format(private_key['stdout']))
    sh.run_on_all('chmod 600 $HOME/.ssh/id_rsa')
    sh.run_on_all('sudo chown root:root ${HOME}/ssh_container/config')
    sh.run_on_all('printf "#!/bin/bash\n" >> $HOME/.ssh/mpicont.sh')
    sh.run_on_all('printf "echo \\"entering container\\"\n" >> $HOME/.ssh/mpicont.sh')
    sh.run_on_all('printf "docker exec mpicont /bin/bash -c \\"\\$SSH_ORIGINAL_COMMAND\\"\n" >> $HOME/.ssh/mpicont.sh')
    sh.run_on_all('chmod +x $HOME/.ssh/mpicont.sh')
    sh.run_on_all('printf "command=\\"bash $HOME/.ssh/mpicont.sh\\"" >> $HOME/.ssh/authorized_keys')
    sh.run_on_all('printf ",no-port-forwarding,no-agent-forwarding,no-X11-forwarding " >> $HOME/.ssh/authorized_keys')
    sh.run_on_all('printf "{0}\n" >> $HOME/.ssh/authorized_keys'.format(public_key['stdout']))

class Notebook(object):

    def __init__(self, clush, notebook_port = 8890, tensorboard_port = 6008):
        self.notebook_port = notebook_port
        self.tensorboard_port = tensorboard_port
        self.clush = clush
        self.socket = None
        self.forward_port()

    def forward_port(self):
        host = self.clush.public_ips[0]
        self.socket = "tf-socket-{}".format(int(time() * 10000))
        start_port_forwarding = "ssh -i {0} -o StrictHostKeyChecking=no " \
                                "-M -S {1} -fNT -L {3}:localhost:8888 " \
                                "-L {4}:localhost:6006 ubuntu@{2}".format(self.clush.key,
                                                                          self.socket,
                                                                          host,
                                                                          self.notebook_port,
                                                                          self.tensorboard_port)
        os.system(start_port_forwarding)
        print(os.system("ssh -S {} -O check ubuntu@{}".format(self.socket, host)))

    def disconnect(self):
        host = self.clush.public_ips[0]
        print(os.system("ssh -S {} -O exit ubuntu@{}".format(self.socket, host)))

    def get_token(self, container_name='mpicont'):
        token = self.clush.run_on_master("docker exec {} bash -c \"jupyter notebook list\"" \
                                          .format(container_name))['stdout'].split('token=')[1].split()[0]
        return "http://localhost:{}/?token={}".format(self.notebook_port, token)



