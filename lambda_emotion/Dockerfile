FROM amazon/aws-lambda-python:3.8
# Set AWS credentials
ENV AWS_ACCESS_KEY_ID="AKIAZNQEHJKK2XKDESWW"
ENV AWS_SECRET_ACCESS_KEY="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"
# Install unzip, AWS CLI, and upgrade pip in one RUN command
RUN yum install unzip -y && \
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    /var/lang/bin/python3.8 -m pip install --upgrade pip && \
    yum install git -y && \
    pip install torch==2.0.1+cpu torchvision==0.15.2+cpu --index-url https://download.pytorch.org/whl/cpu && \
    git clone https://github.com/TeamPmu/Pmu_Data && \
    pip install -r Pmu_Data/lambda_emotion/requirement.txt && \
    cp Pmu_Data/lambda_emotion/emotionclassifier.py /var/task/

# Define the lambda_handler function to execute when the Lambda function is invoked
CMD ["emotionclassifier.lambda_handler"]
