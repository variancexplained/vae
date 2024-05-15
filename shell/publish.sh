echo $'\nPublishing vaebook...'
jb build -W -n --keep-going vaebook/
ghp-import -n -p -f vaebook/_build/html