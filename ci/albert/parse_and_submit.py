import os, sys
import boto3
import re

def regex_extract(text, pattern):
    m = re.search(pattern, text)
    print(text)
    if m:
        found = m.group(1)
    return found

def extract_result(log_abspath):
    result = {}
    loss = 100
    mlm = 0
    sop = 0
    with open(log_abspath, 'r') as log:
        for line in log:
            if 'MLM_acc: ' in line:
                temp_mlm = float(regex_extract(line, 'MLM_acc: ([-+]?\d*\.\d+|\d+)'))
                mlm = max(mlm, temp_mlm)
            if 'SOP_acc: ' in line:
                temp_sop = float(regex_extract(line, 'SOP_acc: ([-+]?\d*\.\d+|\d+)'))
                sop = max(sop, temp_sop)
            if 'Loss: ' in line:
                temp_loss = float(regex_extract(line, 'Loss: ([-+]?\d*\.\d+|\d+)'))
                loss = min(loss, temp_loss)
            if 'Training seconds: ' in line:
                time = float(regex_extract(line, 'Training seconds: ([-+]?\d*\.\d+|\d+)'))
                result['time'] = time
    result['mlm'] = mlm
    result['sop'] = sop
    result['loss'] = loss
    return result

def upload_metrics(parsed_results, num_gpus, batch_size, instance_type, platform):
    client = boto3.client('cloudwatch')
    print(parsed_results['loss'])
    client.put_metric_data(
      Namespace='ModelOptimization',
      MetricData=[
        {
          'MetricName': 'Loss',
          'Value': parsed_results['loss'],
          'Dimensions': [
              {
                  'Name': 'Model',
                  'Value': 'ALBERT'
              },
              {
                  'Name': 'Platform',
                  'Value': str(platform)
              },
              {
                  'Name': 'Instance Type',
                  'Value': str(instance_type)
              },
              {
                  'Name': 'Num of GPUs',
                  'Value': 'GPUs:' + str(num_gpus)
              },
              {
                  'Name': 'Batch Size',
                  'Value': 'Batch Size:' + str(batch_size)
              }
          ]
        }
      ]
    )
    print(parsed_results['mlm'])
    client.put_metric_data(
      Namespace='ModelOptimization',
      MetricData=[
        {
          'MetricName': 'MLM Accuracy',
          'Value': parsed_results['mlm'],
          'Dimensions': [
              {
                  'Name': 'Model',
                  'Value': 'ALBERT'
              },
              {
                  'Name': 'Platform',
                  'Value': str(platform)
              },
              {
                  'Name': 'Instance Type',
                  'Value': str(instance_type)
              },
              {
                  'Name': 'Num of GPUs',
                  'Value': 'GPUs:' + str(num_gpus)
              },
              {
                  'Name': 'Batch Size',
                  'Value': 'Batch Size:' + str(batch_size)
              }
          ]
        }
      ]
    )
    print(parsed_results['sop'])
    client.put_metric_data(
      Namespace='ModelOptimization',
      MetricData=[
        {
          'MetricName': 'SOP Accuracy',
          'Value': parsed_results['sop'],
          'Dimensions': [
              {
                  'Name': 'Model',
                  'Value': 'ALBERT'
              },
              {
                  'Name': 'Platform',
                  'Value': str(platform)
              },
              {
                  'Name': 'Instance Type',
                  'Value': str(instance_type)
              },
              {
                  'Name': 'Num of GPUs',
                  'Value': 'GPUs:' + str(num_gpus)
              },
              {
                  'Name': 'Batch Size',
                  'Value': 'Batch Size:' + str(batch_size)
              }
          ]
        }
      ]
    )
    print(parsed_results['time'])
    client.put_metric_data(
      Namespace='ModelOptimization',
      MetricData=[
        {
          'MetricName': 'Training time',
          'Value': parsed_results['sop'],
          'Dimensions': [
              {
                  'Name': 'Model',
                  'Value': 'ALBERT'
              },
              {
                  'Name': 'Platform',
                  'Value': str(platform)
              },
              {
                  'Name': 'Instance Type',
                  'Value': str(instance_type)
              },
              {
                  'Name': 'Num of GPUs',
                  'Value': 'GPUs:' + str(num_gpus)
              },
              {
                  'Name': 'Batch Size',
                  'Value': 'Batch Size:' + str(batch_size)
              }
          ]
        }
      ]
    )



if __name__ == '__main__':
    abspath = os.path.join(os.getcwd(), sys.argv[1])
    parsed_results = extract_result(abspath)
    print(parsed_results['loss'])
    print(sys.argv[2])
    print(sys.argv[3])
    print(sys.argv[4])
    if parsed_results['loss'] < 100:
        upload_metrics(parsed_results, sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    else:
        print('nothing being parsed')