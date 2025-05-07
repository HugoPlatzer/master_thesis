# Useful commands for working with experiments

## Copy all config.json files of experiment directories from old to new (including directory structure)
```
b=baseline; n=new; find "$b" -type d -name '*k' -print0 | while IFS= read -r -d '' d; do   r=${d#"$b"/};   mkdir -p "$n/$r";   cp "$d/config.json" "$n/$r/"; done
```

## Edit a single JSON config file
```
jq '.training_params.strip_intermediate = false' reverse/add/10digits/1k/config.json | sponge reverse/add/10digits/1k/config.json
```

## Edit all JSON files recursively, in some directory
```
find reverse/add/3digits/ -type f -name 'config.json' | xargs -I {} sh -c 'jq ".model_params.n_positions=24" --indent 4 {} | sponge {}'
find reverse/ -type f -name 'config.json' | xargs -I {} sh -c 'jq ".sampler.params.intermediate_steps=\"reverse\"" --indent 4 {} | sponge {}'
```

# Find all experiment dirs of some kind, apply some json transform to them
```
for f in $(find baseline/ -type d -name 1k); do conffile=$f/config.json; jq '.training_params.logging_steps=1' $conffile --indent 4 | sponge $conffile; done
```

## Run all experiments in some directory (assumes no checkpoint config.json files)

```
find experiments/intermediate_steps/reverse/ -name '*k' -type d | sort -r | xargs -n 1 echo "running experiment"
find experiments/intermediate_steps/reverse/ -name '*k' -type d | sort -r | xargs -n 1 python run_experiment.py
```

## Clean up extra files after experiments have completed in some directory

```
find . -name 'checkpoint-*' -exec rm -f {}/optimizer.pt {}/rng_state.pth {}/scheduler.pt {}/trainer_state.json {}/training_args.bin \;
```

## Create tgz file of some experiments dir
```
tar czf experiments.tgz experiments/intermediate_steps/reverse/
```

## Transfer experiments.tgz from remote server to localhost using ssh
```
sftp -P PORT root@IP_ADDR:/root/master_thesis/code/experiments.tgz .
```

## Unpack tgz file of experiments
```
tar tf ~/experiments.tgz
tar xvf ~/experiments.tgz
```
