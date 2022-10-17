while read line; do
    start msedge --new-tab "$line"
done < links.txt