import { useEffect, useState } from 'react';
import CopiedToast from '../components/CopiedToast';
import NavBar from '../components/Navbar';
import SearchBar from '../components/SearchBar';
import SnippetsContainer from '../components/SnippetsContainer';
import ISnippet from '../lib/ISnippet';
import useStore from '../lib/store';
// import data from '../lib/data';

export default function Index() {
  const numFilteredSnippets = useStore((state) => state.numFilteredSnippets);
  const setNumFilteredSnippets = useStore(
    (state) => state.setNumFilteredSnippets
  );

  const [copied, setCopied] = useState<boolean>(false);
  const [data, setData] = useState<ISnippet[]>([]);
  const [isData, setIsData] = useState<boolean>(false);

  useEffect(() => {
    const fetchData = async () => {
      const data = (await import('../lib/data')).default;
      setData(data);
      setNumFilteredSnippets(data.length);
      // await new Promise((r) => setTimeout(r, 2000));
      setIsData(true);
    };
    fetchData();
  }, []);

  return (
    <>
      <NavBar />
      <SearchBar />
      {isData ? (
        <SnippetsContainer snippets={data} setCopied={setCopied} />
      ) : (
        <div className="w-full text-center text-white">Loading Cards ...</div>
      )}
      {/* <SnippetsContainer snippets={data} setCopied={setCopied} /> */}
      <CopiedToast
        text={'Code Successfully copied to clipboard!'}
        copied={copied}
      />
    </>
  );
}
