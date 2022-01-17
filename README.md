# infoseq
Information theory on sequences (probably mostly language modeling and transformers)

## Downloading Project Gutenberg

See the [Project Gutenberg page on robot access](https://www.gutenberg.org/policy/robot_access.html) for more information about downloading.

For now we will just get txt file versions of the English books.

```bash
    wget -w 2 -m -H "http://www.gutenberg.org/robot/harvest?filetypes[]=txt&langs[]=en"
```

## TODO

Currently working on tokenization and BPE stuff.

Need to do:
* Add a way to save a tokenization to a file and load it back
* Preprocess KJV tokenization to BPE
* Make a visualization of the tokenization with graphviz