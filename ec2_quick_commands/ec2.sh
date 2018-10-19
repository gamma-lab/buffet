#
# A non-system administrator's guide
# Quick commands to manage working with remote EC2 server
#
# Instructions:
#  a) copy this file to ~/.bashrc or ~/.bash_profile
#  b) update the values
#  c) modify `--exclude` option in `ec2.sync` to suite your requirements
#  d) install sshfs https://osxfuse.github.io/
#

######## Variables #########

# Absolute path to the private key that has access to EC2 (you must have downloaded this
# at the time of setting up EC2 instance)
REMOTE_EC2_KEY="~/Documents/projects/dstc7/.ssh/chatbot.pem"

# user name and IP address of EC2 instance. 
# NOTE: If the IP is not static, it will change every time you restart the instance
REMOTE_EC2_ADDR="ubuntu@ec2-54-187-222-158.us-west-2.compute.amazonaws.com"

# Open this directory in EC2 after login
# If this PATH does not exist, EC2 falls back to the home directory ~
REMOTE_EC2_LOGIN_DIR="/home/ubuntu/"

# Your project directory relative to `REMOTE_EC2_LOGIN_DIR` to mount to local computer. 
# NOTE: This can be empty, and the REMOTE_EC2_LOGIN_DIR is assumed as project directory.
#
# `$ ec2.sync` will copy all the files in `LOCAL_PROJECT_DIR` to `REMOTE_PROJECT_DIR`
REMOTE_PROJECT_DIR="dstc7"

# Mount your `REMOTE_PROJECT_DIR` to this directory in your local computer
# NOTE: this must be absolute path
LOCAL_MOUNT_DIR="~/Documents/EC2/"

# Your project directory on local computer, that you want to copy to EC2
#
# `$ ec2.sync` will copy all the files in `LOCAL_PROJECT_DIR` to `REMOTE_PROJECT_DIR`
LOCAL_PROJECT_DIR="~/Documents/projects/dstc7/"

# Setup a SSH tunnel on this port number with local computer and EC2 instance
TUNNEL_PORT=8888

######## Commands #########

# execute `$ ec2.tunnel`
alias ec2.tunnel="ssh -i \"$REMOTE_EC2_KEY\" -N -L :$TUNNEL_PORT:localhost:$TUNNEL_PORT $REMOTE_EC2_ADDR"

# execute `$ ec2.login`
alias ec2.login="ssh -i \"$REMOTE_EC2_KEY\" $REMOTE_EC2_ADDR"

# execute `$ ec2.mount`
alias ec2.mount="sshfs -o volname=EC2 $REMOTE_EC2_ADDR:$REMOTE_EC2_LOGIN_DIR $LOCAL_MOUNT_DIR -o IdentityFile=\"$REMOTE_EC2_KEY\""

# execute `$ ec2.sync`
alias ec2.sync="rsync -arv --exclude=\".ssh\" --exclude=\".git\" --exclude=venv --exclude=\"*.pyc\" --exclude=\"train.*.txt\" $LOCAL_PROJECT_DIR $LOCAL_MOUNT_DIR/$REMOTE_PROJECT_DIR"
