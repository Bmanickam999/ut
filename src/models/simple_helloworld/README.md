# Simple Hello World

This package preserves the original lightweight demonstration separately from
the one-million-example leadership run in `models/helloworld`.

Both implementations use 128 examples, two epochs, batch size 16, SGD at
`0.001`, and a constant learning rate:

- `before/helloworld.py` intentionally combines model and infrastructure code.
- `new/helloworld.py` contains only the model contract; `generic_trainer` owns
  optimization, execution, logging, and checkpoints from `new/config.yaml`.

Run both sides and generate a generic visual comparison:

```bash
./src/models/simple_helloworld/run_comparison.sh
```

The existing `models/helloworld` package remains the one-million-example,
five-epoch leadership demonstration.
