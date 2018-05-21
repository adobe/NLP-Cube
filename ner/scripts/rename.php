<?php
$lang=$argv[1];
$uplang=strtoupper($lang);
$command="mv models/$lang/model_$uplang-best-fscore.network models/$lang/model-best-fscore.network";
echo "$command\n";
exec ($command);
$command="mv models/$lang/model_$uplang-last.network models/$lang/model-last.network";
echo "$command\n";
exec ($command);
$command="mv models/$lang/model_$uplang.encodings models/$lang/model.encodings";
echo "$command\n";
exec ($command);

?>
