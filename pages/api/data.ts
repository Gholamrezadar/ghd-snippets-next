// Next.js API route support: https://nextjs.org/docs/api-routes/introduction

import data from '../../lib/data';

export default (req: any, res: any) => {
  // Open Chrome DevTools to step through the debugger!
  // debugger;
  res.status(200).json(data);
};
