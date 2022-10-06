import { useEffect, useState } from 'react';
import CopiedToast from '../components/CopiedToast';
import NavBar from '../components/Navbar';
import SearchBar from '../components/SearchBar';
import SnippetsContainer from '../components/SnippetsContainer';
import data from '../lib/data';

export default function Index() {
  const [tags, setTags] = useState<string[]>([]);
  const [copied, setCopied] = useState<boolean>(false);

  useEffect(() => {
    // use Set to get the unique 'tags' from 'data'
    let tags = new Set(
      data.flatMap((item) => {
        return item.tags;
      })
    );

    console.log('tags', tags);

    // save the extracted tags in the 'tags' state
    setTags([...tags]);
  }, []);

  return (
    <>
      <NavBar />
      <SearchBar tags={tags} />
      <SnippetsContainer snippets={data} setCopied={setCopied} />
      <CopiedToast
        text={'Code Successfully copied to clipboard!'}
        copied={copied}
      />
    </>
  );
}
