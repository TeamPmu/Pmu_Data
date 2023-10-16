FROM amazon/aws-lambda-python:3.8

# Set AWS credentials
ENV AWS_ACCESS_KEY_ID="AKIAZNQEHJKK2XKDESWW"
ENV AWS_SECRET_ACCESS_KEY="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"

# Install unzip
RUN yum install unzip -y

# Install AWS CLI
USER root
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
RUN unzip awscliv2.zip
RUN ./aws/install

RUN aws s3 cp s3://modelpmu/model_1 /var/task/model_1 --recursive

RUN /var/lang/bin/python3.8 -m pip install --upgrade pip


RUN yum install git -y


RUN git clone https://github.com/timesler/facenet-pytorch
RUN git clone https://github.com/TeamPmu/Pmu_Data


# install packages
RUN pip install -r Pmu_Data/lambda_emotion/requirement.txt


# /var/task/ 경로로 실행파일 복사
RUN cp Pmu_Data/lambda_emotion/emotionclassifier.py /var/task/

# 실행 시 lambda_function.py의 lambda_handler 함수를 실행시킴을 정의
CMD ["emotionclassifier.lambda_handler"]