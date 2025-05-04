# Useful commands for working with experiments

## Copy all config.json files of experiment directories from old to new (including directory structure)
```
src_dir="new"
dest_dir="new"
find "$src_dir" -type d -name '*k' -exec find {} -maxdepth 1 -type f -name 'config.json' \; | while read -r file; do
    dest_path="$dest_dir/$(echo "$file" | sed "s#$src_dir/##")"
    mkdir -p "$dest_path"
    cp "$file" "$dest_path"
done
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

## Run all experiments in some directory (assumes no checkpoint config.json files)

```
find experiments/intermediate_steps/reverse/ -name '*k' -type d | sort -r | xargs echo "running experiment"
find experiments/intermediate_steps/reverse/ -name '*k' -type d | sort -r | xargs python run_experiment.py
```
