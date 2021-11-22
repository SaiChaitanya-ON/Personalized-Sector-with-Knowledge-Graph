import json
import boto3
import sys
import time

if len(sys.argv) == 1:
  with open("emr/cluster_id.json") as f:
      cluster_id = json.load(f)["ClusterId"]
else:
  cluster_id = sys.argv[1]

client = boto3.client('emr', region_name='eu-west-1')

MAX_TRIES = 100
DELAY = 30
n_attempts = 0

while n_attempts < MAX_TRIES:
  response = client.describe_cluster(ClusterId=cluster_id)
  if "MasterPublicDnsName" in response["Cluster"]:
    master_dns = response["Cluster"]["MasterPublicDnsName"]
    break
  n_attempts += 1
  time.sleep(DELAY)


livy_config = f'''
{{
  "kernel_python_credentials" : {{
    "username": "",
    "password": "",
    "url": "http://{master_dns}:8998",
    "auth": "None"
  }},

  "kernel_scala_credentials" : {{
    "username": "",
    "password": "",
    "url": "http://{master_dns}:8998",
    "auth": "None"
  }},
  "kernel_r_credentials": {{
    "username": "",
    "password": "",
    "url": "http://{master_dns}:8998"
  }},

  "logging_config": {{
    "version": 1,
    "formatters": {{
      "magicsFormatter": {{ 
        "format": "%(asctime)s\\t%(levelname)s\\t%(message)s",
        "datefmt": ""
      }}
    }},
    "handlers": {{
      "magicsHandler": {{ 
        "class": "hdijupyterutils.filehandler.MagicsFileHandler",
        "formatter": "magicsFormatter",
        "home_path": "~/.sparkmagic"
      }}
    }},
    "loggers": {{
      "magicsLogger": {{ 
        "handlers": ["magicsHandler"],
        "level": "DEBUG",
        "propagate": 0
      }}
    }}
  }},

  "wait_for_idle_timeout_seconds": 15,
  "livy_session_startup_timeout_seconds": 120,

  "fatal_error_suggestion": "The code failed because of a fatal error:\\n\\t{{}}.\\n\\nSome things to try:\\na) Make sure Spark has enough available resources for Jupyter to create a Spark context.\\nb) Contact your Jupyter administrator to make sure the Spark magics library is configured correctly.\\nc) Restart the kernel.",

  "ignore_ssl_errors": false,

  "session_configs": {{
    "driverMemory": "1000M",
    "executorCores": 2
  }},

  "use_auto_viz": true,
  "coerce_dataframe": true,
  "max_results_sql": 2500,
  "pyspark_dataframe_encoding": "utf-8",
  
  "heartbeat_refresh_seconds": 30,
  "livy_server_heartbeat_timeout_seconds": 0,
  "heartbeat_retry_seconds": 10,

  "server_extension_default_kernel_name": "pysparkkernel",
  "custom_headers": {{}},
  
  "retry_policy": "configurable",
  "retry_seconds_to_sleep_list": [0.2, 0.5, 1, 3, 5],
  "configurable_retry_policy_max_retries": 8
}}
'''

with open("/root/.sparkmagic/config.json", "w+") as f:
    f.write(livy_config)
