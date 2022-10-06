const Hash = (a: string) => {
  var hash = 0;
  if (a.length == 0) return hash;
  for (let x = 0; x < a.length; x++) {
    let ch = a.charCodeAt(x);
    hash = (hash << 5) - hash + ch;
    hash = hash & hash;
  }
  return hash;
};

export default Hash;
