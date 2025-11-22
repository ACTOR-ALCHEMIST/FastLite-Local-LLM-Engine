#!/bin/bash

# Server Details
SERVER_IP="34.242.203.50"
USER="user"
KEY="user.pem"

# Ensure key has correct permissions
chmod 600 $KEY

echo "Syncing files to server..."
# Upload inferencePipeline and run.py
scp -i $KEY -r ./inferencePipeline ./run.py $USER@$SERVER_IP:~/

echo "Running tests on server..."
# Execute run.py on server
ssh -i $KEY $USER@$SERVER_IP "python3 -u run.py"
