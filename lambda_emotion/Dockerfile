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
<<<<<<< HEAD
    aws s3 cp s3://pmu-bucket/facenet_pytorch/ /var/task/ --recursive && \
=======
>>>>>>> 2a49295e6902b08dd12757023860cfec334178f2
    git clone https://github.com/TeamPmu/Pmu_Data && \
    pip install -r Pmu_Data/lambda_emotion/requirement.txt && \
    cp Pmu_Data/lambda_emotion/emotionclassifier.py /var/task/

# Define the lambda_handler function to execute when the Lambda function is invoked
CMD ["emotionclassifier.lambda_handler"]
