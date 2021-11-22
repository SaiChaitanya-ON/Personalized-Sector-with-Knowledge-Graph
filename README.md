# onaiml [![CircleCI](https://circleci.com/gh/OakNorthAI/onai.ml.svg?style=svg&circle-token=a4059ed07bb9218e4f81ed9bb03084d1b240d861)](https://circleci.com/gh/OakNorthAI/onai.ml)
ONAI Machine Learning Library


## Proposed Workflow

Developing production-ready ML models is a highly iterative process of trial and error and prototyping
of ideas, that will eventually end up as production-ready code. This repo aims to be a meeting point
for both aspects of ML development.

To this end, we have the [notebooks/](https://github.com/OakNorthAI/acornML/tree/develop/notebooks/) dir
where we can hold our prototyping notebooks.
We also have our internal ml library under [onaiml/](https://github.com/OakNorthAI/onai.ml/tree/master/onai/ml).
This should hold our production-ready code, together with appropriate unit tests. The idea is that
we quickly prototype ideas in `notebooks/`, and once we're happy with what we have, we tidy up the code
and place it under `onai/ml/`. As far as PRs go, we should be minimally critical of things getting
merged to `notebooks/` and maximally critical of things getting merged to `onai/ml/`.

### Consistency of Python environment

Making sure that what we develop as notebooks works the same when packaged for deployment is important.
To ensure this consistency we follow the example of `marmot` and use poetry to manage our
python environment. This generates a textual description of the python environment which we
can check into source control.

To avoid other system-specific inconsistencies, and make sure the workflow is WFH friendly, we recommend you run everything in garden with poetry. You can also run the code on your own laptop if you initialise your local environment correctly, but the consistency of results might be questionable due to slight difference between environments, e.g. Mac OS v.s. Linux kernels. 

### Nice, now how do I actually work on this with garden?

The above also makes it relatively easy to develop for this package:

1. Follow [these](https://acornlab.atlassian.net/wiki/spaces/PROD/pages/689831962/Poetry+IDEs) instructions to set up poetry locally and with your favourite IDE. The instructions are for `marmot`, but are easily adaptable for here
2. Ensure you have the `garden == 0.12.6` and `k9s` installed
3. Ensure you have [devspace](https://github.com/devspace-cloud/devspace#1-install-devspace) installed
4. Ensure you have [ml-tools](https://github.com/OakNorthAI/ml-tools) installed
5. Ensure you have [oak](https://github.com/OakNorthAI/oak)
5. In linux system, ensure you have openvpn-systemd-resolved and nfs-common installed.
   `sudo apt install openvpn-systemd-resolved nfs-common`
6. `setup_env init-aws` if this is the first time you have not set-up aws yet
   * 🍏 On a mac you need to do some workarounds to make bdb work:
      ```
      brew install Berkeley-db@4
      export YES_I_HAVE_THE_RIGHT_TO_USE_THIS_BERKELEY_DB_VERSION=1
      export BERKELEYDB_DIR=/usr/local/Cellar/berkeley-db@4/4.8.30
      ```
7. `make run-dev` to build and deploy an experimental platform to your own specific k8s namespace.
8.  Start a synchorniser: this will sync your local code to the remote pod bidirectionally.
   ```
   # set the namespace, normally just your name-surname
   export YOUR_NAMESPACE=$(cat garden.env|grep ns=|sed 's/.*=//')
   # start
   devspace sync --container-path /acornML --local-path ./ --label-selector=service=ml-dev --namespace $YOUR_NAMESPACE --exclude ".garden/*" --exclude "dist/*" --exclude "*.pyc" --exclude ".git/*"
   ```
9. Connect to vpn by `sudo setup_env init-dev-env`.
   * 🍏 On a mac you will have to use this [solution](https://apple.stackexchange.com/questions/388236/unable-to-create-folder-in-root-of-macintosh-hd) to create the `/mnt` folder that the script needs
10. Go wild! You can use this command to view your notebook server. `open http://ml.${YOUR_NAMESPACE}.svc.cluster.local:8888/`. You can be rest assured that
   * the python env it uses is the one specified by poetry and that it also has `onaiml` installed.
   * if you edit the `onaiml` package in your favourite editor/IDE and the changes will be reflected in the container.

### Initialise local python environment
If you want to run python scripts in your local environment, you have to initiliase your local virtualenv properly with poetry.

To do that, you should 
1. Initialise a virtualenv, that virtualenv should stay outside of the source folder.
2. activate the virtualenv
3. go to your source root
4. `oak pypi-login`
5. `poetry install`

### Updating the python environment

As we progress with our endeavours, we will almost certainly find that our dependencies need updating.
This is easy with poetry. On your local laptop, in the source root, do:
0. `oak pypi-login`
1. `poetry add <your dep>`.

Note that you can use the same procedure with garden exec to update your dependency remotely, e.g. when your network is extremely poor, and sync back the changed poetry.lock and pyproject.toml w/ devspace.

pyspark is installed w/ our own customised package so you should avoid updating it.

Other dependencies are installed in `ml.base.Dockerfile `directly.

### Testing

As we progress our code from notebooks to the actual package, it is good practice to add some unit
tests. Their primary purpose should be to ensure correctness of data-munging and other processing
steps without strictly casting any opinion on how the models perform (this needs discussion).

These tests should live under an appropriate subdirectory of
[onaiml/tests/](https://github.com/OakNorthAI/acornML/tree/develop/onaiml/tests) (usually the same
as the submodule they are testing).

To run your tests, you have a number of options:
* `make dev-test` : this runs test in an existing garden deployment. To select which test to run, prepend `PYTEST_ARG=path/to/test/to/run`
* `make bash` allow you to enter the running garden deployment and run your tests manually using `pytest`

### Release

1. Bump the package version in the `pyproject.toml` file, using `poetry version patch|minor|major`.
1. Tag the commit on master with the version number (`git tag ``poetry version | cut -d' ' -f2-`` && git push --tags`).
1. Create a PR against master. When the PR gets merged, the package will be published for you.


### Version Semantics

#### Code Version Semantics
We follow the standard here

https://semver.org/

1. MAJOR version when you make incompatible API changes,
2. MINOR version when you add functionality in a backwards compatible manner, and
3. PATCH version when you make backwards compatible bug fixes.

#### ML base image version semantics
We follow exactly the same semantics as code version.

There is a base image release along w/ each code release. This is triggered manually and the version follows the style
of code version, i.e. X.Y.Z.

There are also ML base images that are generated on every Friday and tagged w/ hash key and mlbase:latest_dev.

They are development images and should not be used in production environment.

### Running on SageMaker

To train some of the computational intensive models, we need to run them on GPU cluster. At the moment, we can send our training job to sagemaker. To make sure the consistency of the environment we run our experiment on the local laptop, garden cluster, and sagemaker cluster, we used the same set of dockerfiles to build all these images. Before you embark on the journey, make sure you have read https://oaknorth.atlassian.net/wiki/spaces/ML/pages/1483046948/Sagemaker+Onboarding already. 

An example to start understanding how sagemaker in this repo works is to look at the industry classification folder. The steps are 

1. Implement a trainer script, e.g. onai/ml/industry/experimental/mm/model.py and make sure it can work locally (or on garden cluster)
2. Package the whole onai.ml repo by 
```
rm -rf dist && poetry build &&  aws s3 cp dist/*tar.gz $S3_DEV_HOME/code/onaiml.tar.gz
```
3. Write a sagemaker script, e.g. onai/ml/industry/experimental/mm/sm.py, so that it will send a aws training job request.

Once you do all this and execute the sagemaker script, you probably will see that sagemaker is complaining the training job image is not found. The reason is that you have not built (and push it yet). Use the following command to initialise the image.

```
make init-sm-image
```

### Troubleshooting
If you see

```
401 Client Error: Unauthorized for URL: https://oaknorth-xxxxxxxxx ....
```

You have not autheniticated your poetry correctly to our private pypi repo. 

You should run ``oak pypi-login`` that before you do anything w/ poetry in this repo.

### Running on EMR

To run things at scale, we need to move to the cloud. Amazon EMR provides a (reasonably) good
out-of-the-box spark cluster which we can connect to. To do this, you need to run the following.

```
make run-dev
make create_dev_emr_cluster
```

This will automatically spin up an EMR cluster for you and set up the running docker container to
connect to it. To run stuff against it, simply select PySpark as your kernel.

One can specify instances for master and core (worker) nodes as follows:

`make create_dev_emr_cluster INSTANCES={<master>,<core>}`
   like
`make create_dev_emr_cluster INSTANCES=m4.xlarge,t2.nano`

It should create a cluster named `MLEng - <firstname>` (it's based on `whoami`).

If you already have an existing EMR cluster (from previous attempts), you can run

```
make run-dev
make create_dev_emr_cluster CLUSTER_ID={cluster id from aws}
```

This will configure your running docker container to connect to the existing EMR cluster.
You can safely assume that the cluster has all the deps specified in the `pyproject.toml` file that
was present in the repo when the cluster was spun up initially.

In terms of scaling up your cluster, please use the AWS console. These clusters should be small-ish
dev clusters to help us develop spark jobs to later submit to a proper large-scale cluster.
