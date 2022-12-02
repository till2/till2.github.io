/* REWARD HANDLING FOR MDP */

function updateAccReward(reward){

  /* load */
  var accumulated_reward = localStorage.getItem("accumulated_reward");

  if(!isNumeric(accumulated_reward)){
    accumulated_reward = 0;
  }

  document.getElementById("reward_field").innerHTML = parseInt(accumulated_reward) + parseInt(reward);

  /* save */
  var accumulated_reward = document.getElementById("reward_field").innerHTML;
  localStorage.setItem("accumulated_reward", accumulated_reward);
}

function resetMDP(){
  localStorage.setItem("accumulated_reward", 0); /* reset accumulated reward to 0*/
  localStorage.clear();
  document.getElementById("reward_field").innerHTML = 0; /* reset displayed reward*/
}

function isNumeric(str) {
  if (typeof str != "string") return false // we only process strings!  
  return !isNaN(str) && // use type coercion to parse the _entirety_ of the string (`parseFloat` alone does not do this)...
         !isNaN(parseFloat(str)) // ...and ensure strings of whitespace fail
}