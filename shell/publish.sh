echo $'\nPublishing vaebook...'
jb build --all vaebook
ghp-import -n -p -f vaebook/_build/html