
var TEAMS = {
  NYY:'New York Yankees', BAL:'Baltimore Orioles', BOS:'Boston Red Sox', TBR:'Tampa Bay Rays',
  TOR:'Toronto Blue Jays', LAD:'Los Angeles Dodgers', SDP:'San Diego Padres', ARI:'Arizona D-backs',
  SFG:'San Francisco Giants', COL:'Colorado Rockies', PHI:'Philadelphia Phillies', ATL:'Atlanta Braves',
  CLE:'Cleveland Guardians', HOU:'Houston Astros', MIL:'Milwaukee Brewers'
};
function go(id){
  document.querySelectorAll('.tab').forEach(function(t){ t.classList.toggle('is-on', t.dataset.go === id); });
  document.querySelectorAll('.screen').forEach(function(s){ s.classList.toggle('is-on', s.id === id); });
  window.scrollTo({top:0, behavior:'smooth'});
}
document.querySelectorAll('.tcard').forEach(function(c){
  c.addEventListener('click', function(){
    var ab = c.dataset.team;
    document.getElementById('mtAb').textContent = ab;
    document.getElementById('mtName').textContent = TEAMS[ab] || ab;
    document.getElementById('enter').style.display = 'none';
    document.getElementById('app').classList.add('is-on');
    go('war');
  });
});
document.getElementById('btnChange').addEventListener('click', function(){
  document.getElementById('app').classList.remove('is-on');
  document.getElementById('enter').style.display = 'block';
  window.scrollTo({top:0});
});
document.querySelectorAll('[data-go]').forEach(function(el){
  el.addEventListener('click', function(){ go(el.dataset.go); });
});
document.querySelectorAll('[data-open]').forEach(function(el){
  el.addEventListener('click', function(){ go(el.dataset.open); });
});
document.querySelectorAll('.chip, .lg-btn').forEach(function(el){
  el.addEventListener('click', function(){
    if (el.disabled) return;
    var sel = el.classList.contains('chip') ? '.chip' : '.lg-btn';
    el.parentElement.querySelectorAll(sel).forEach(function(s){ s.classList.toggle('is-on', s === el); });
  });
});

// Streamlit의 각 페이지가 body[data-screen]으로 초기 화면을 지정한다.
var initialScreen = document.body.dataset.screen || 'home';
if (initialScreen === 'home') {
  document.getElementById('enter').style.display = 'block';
  document.getElementById('app').classList.remove('is-on');
} else {
  document.getElementById('enter').style.display = 'none';
  document.getElementById('app').classList.add('is-on');
  go(initialScreen);
}
