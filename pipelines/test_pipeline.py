"""Trivial test pipeline to verify DSPA is functional."""
from kfp import dsl, compiler


@dsl.component(base_image="registry.access.redhat.com/ubi9/python-311:latest")
def hello(message: str) -> str:
    print(f"Hello from DSPA: {message}")
    return message


@dsl.pipeline(name="dspa-smoke-test", description="Trivial pipeline to verify DSPA works")
def smoke_test():
    hello(message="Phase 2 pipeline server is functional")


if __name__ == "__main__":
    compiler.Compiler().compile(smoke_test, "pipelines/test_pipeline.yaml")
    print("Compiled to pipelines/test_pipeline.yaml")
