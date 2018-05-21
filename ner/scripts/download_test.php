<?php
$lang=$argv[1];
$langup=strtoupper($lang);
$command="wget https://gitlab.com/parseme/sharedtask-data/raw/master/1.1/$langup/test.blind.cupt -O corpus/test/$langup/test.blind.cupt";
exec($command);
?>
