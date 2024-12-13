# Post-Flight Repo Cleanup Checklist

Fix these naming inconsistencies:
- "pretrained" -> pile-dedup
- "recreation" -> reproduction (although funny)
- "proportionate" -> proportioned

- Check for the above in all files and in notebook 03 (esp. the tables)
- Check if all used resources/links are marked/attributed
- Check all HuggingFace links for if they still exist (have been renamed)
- Check if the explained ideas in 03 align with the order of the realized idea scripts
- Scan for any remaining `TODO` comments
- Nuke all remaining tmux sessions
- Make it more clear that *zero-shot* performance is reported with the benchmarks throughout
- export and add to repo: conda env setup