import os, sys
import boto3
import re

def regex_extract(text, pattern):
    m = re.search(pattern, text)
    print(text)
    if m:
        found = m.group()
    return found

def extract_result(log_abspath):
    result = {}
    ap_95_all = 0
    ap_50_all = 0
    ap_75_all = 0
    ap_95_small = 0
    ap_95_medium = 0
    ap_95_large = 0
    with open(log_abspath, 'r') as log:
        for line in log:
            if 'IoU=0.50:0.95 | area=   all | maxDets=100' in line:
                temp_ap_95_all = float(regex_extract(line, '(?<=maxDets\=100\s\]\s\=\s)([-+]?\d*\.\d+|\d+)'))
                ap_95_all = max(ap_95_all, temp_ap_95_all)
            if 'IoU=0.50      | area=   all | maxDets=100' in line:
                temp_ap_50_all = float(regex_extract(line, '(?<=maxDets\=100\s\]\s\=\s)([-+]?\d*\.\d+|\d+)'))
                ap_50_all = max(ap_50_all, temp_ap_50_all)
            if 'IoU=0.75      | area=   all | maxDets=100' in line:
                temp_ap_75_all = float(regex_extract(line, '(?<=maxDets\=100\s\]\s\=\s)([-+]?\d*\.\d+|\d+)'))
                ap_75_all = max(ap_75_all, temp_ap_75_all)
            if 'IoU=0.50:0.95 | area= small | maxDets=100' in line:
                temp_ap_95_small = float(regex_extract(line, '(?<=maxDets\=100\s\]\s\=\s)([-+]?\d*\.\d+|\d+)'))
                ap_95_small = max(ap_95_small, temp_ap_95_small)
            if 'IoU=0.50:0.95 | area=medium | maxDets=100' in line:
                temp_ap_95_medium = float(regex_extract(line, '(?<=maxDets\=100\s\]\s\=\s)([-+]?\d*\.\d+|\d+)'))
                ap_95_medium = max(ap_95_medium, temp_ap_95_medium)
            if 'IoU=0.50:0.95 | area= large | maxDets=100' in line:
                temp_ap_95_large = float(regex_extract(line, '(?<=maxDets\=100\s\]\s\=\s)([-+]?\d*\.\d+|\d+)'))
                ap_95_large = max(ap_95_large, temp_ap_95_large)
            if 'Training seconds: ' in line:
                time = float(regex_extract(line, 'Training seconds: ([-+]?\d*\.\d+|\d+)'))
                result['time'] = time
            
    result['ap_0.95_all'] = ap_95_all
    result['ap_0.50_all'] = ap_50_all
    result['ap_0.75_all'] = ap_75_all
    result['ap_0.95_small'] = ap_95_small
    result['ap_0.95_medium'] = ap_95_medium
    result['ap_0.95_large'] = ap_95_large
    return result

def upload_metrics(parsed_results, num_gpus, batch_size, instance_type, platform):
    client = boto3.client('cloudwatch')
    print(parsed_results['ap_0.95_all'])
    client.put_metric_data(
      Namespace='ModelOptimization',
      MetricData=[
        {
          'MetricName': 'Precision-0.50:0.95-all',
          'Value': parsed_results['ap_0.95_all'],
          'Dimensions': [
              {
                  'Name': 'Model',
                  'Value': 'FasterRCNN'
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
                  'Value': 'Batche Size:' + str(batch_size)
              }
          ]
        }
      ]
    )
    print(parsed_results['ap_0.50_all'])
    client.put_metric_data(
      Namespace='ModelOptimization',
      MetricData=[
        {
          'MetricName': 'Precision-0.50-all',
          'Value': parsed_results['ap_0.50_all'],
          'Dimensions': [
              {
                  'Name': 'Model',
                  'Value': 'FasterRCNN'
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
    print(parsed_results['ap_0.75_all'])
    client.put_metric_data(
      Namespace='ModelOptimization',
      MetricData=[
        {
          'MetricName': 'Precision-0.75-all',
          'Value': parsed_results['ap_0.75_all'],
          'Dimensions': [
              {
                  'Name': 'Model',
                  'Value': 'FasterRCNN'
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
    print(parsed_results['ap_0.95_small'])
    client.put_metric_data(
      Namespace='ModelOptimization',
      MetricData=[
        {
          'MetricName': 'Precision-0.50:0.95-small',
          'Value': parsed_results['ap_0.95_small'],
          'Dimensions': [
              {
                  'Name': 'Model',
                  'Value': 'FasterRCNN'
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
    print(parsed_results['ap_0.95_medium'])
    client.put_metric_data(
      Namespace='ModelOptimization',
      MetricData=[
        {
          'MetricName': 'Precision-0.50:0.95-medium',
          'Value': parsed_results['ap_0.95_medium'],
          'Dimensions': [
              {
                  'Name': 'Model',
                  'Value': 'FasterRCNN'
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
    print(parsed_results['ap_0.95_large'])
    client.put_metric_data(
      Namespace='ModelOptimization',
      MetricData=[
        {
          'MetricName': 'Precision-0.50:0.95-large',
          'Value': parsed_results['ap_0.95_large'],
          'Dimensions': [
              {
                  'Name': 'Model',
                  'Value': 'FasterRCNN'
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
          'Value': parsed_results['time'],
          'Dimensions': [
              {
                  'Name': 'Model',
                  'Value': 'FasterRCNN'
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
    upload_metrics(parsed_results, sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])