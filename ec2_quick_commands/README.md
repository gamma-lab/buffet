# EC2 Quick Commands

Typical workflow

1) Mount EC2 disk to your local computer
2) Copy project files from local computer to EC2
3) Login to EC2 shell and execute project files


## How to use

```
# Login to EC2
$ ec2.login

# Mount EC2 disk to local computer
# ec2.mount

# Setup SSH tunnel at port 8888 (to access Jupyter)
$ ec2.tunnel

# Copy your project files to EC2 (only copies changed files)
# The command does not delete any existing files in EC2. If that's required, you need
# to manually delete those.
$ ec2.sync
```

## How to setup

1) Copy the contents of `ec2.sh` to `~/.bashrc` or `~/.bash_profile`
2) Modify the variables defined in the content (see the comments at the top)
3) Restart your shell


## Requirements

1) Linux or macOS
2) Install Fuse https://osxfuse.github.io/


## Extending this script

1) Check `$ man rsync` to modify `ec2.sync`
2) Check `$ man ssh` to modify `ec2.login`
