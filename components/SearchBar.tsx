import { useEffect, useState } from 'react';
import useStore from '../lib/store';
import CheckBoxItem from './CheckBoxItem';
import FoundSnippetsCounter from './FoundSnippetsCounter';

const SearchBar = () => {
  // console.log('searchbar rerender');
  // Zustand Store
  const filter = useStore((state) => state.filter);
  const setFilter = useStore((state) => state.setFilter);
  const tags = useStore((state) => state.tags);

  const fetchTags = useStore((state) => state.fetchTags);

  const [isData, setIsData] = useState<boolean>(false);
  const [checkBoxListState, setCheckBoxListState] = useState<any>();

  useEffect(() => {
    const fetchData = async () => {
      await fetchTags();
      // await new Promise((r) => setTimeout(r, 1000));
      setIsData(true);
    };
    fetchData();
  }, []);

  useEffect(() => {
    let checkBoxList;
    if (isData) {
      checkBoxList = tags.map((item) => {
        return <CheckBoxItem tag={item} key={item} />;
      });
    } else {
      checkBoxList = <div>Loading Tags ...</div>;
    }
    setCheckBoxListState(checkBoxList);
  }, [isData]);

  return (
    <>
      {/* Search Bar */}
      <div className="flex h-44 md:h-44 w-full items-center justify-center mb-6 md:md-4">
        <div className="flex h-full w-full max-w-4xl flex-col items-center justify-center p-5 dark:text-white">
          {/* Search field */}
          <div className="flex items-center justify-center w-full max-w-2xl mb-2 md:mb-6">
            <div
              tabIndex={0}
              className=" flex form-control m-0 block w-full rounded-full dark:bg-ghd-dark-dark bg-clip-padding px-12 md:py-5 py-2 text-gray-200 transition ease-in-out focus:ring-4 focus:ring-slate-700 focus-ring"
            >
              {/* Search field input */}
              <input
                className="appearance-none focus:outline-none w-full bg-transparent"
                type="text"
                placeholder="Search Snippets ..."
                value={filter}
                onChange={(e) => setFilter(e.target.value)}
              />
              {/* Magnifier svg */}
              <svg
                xmlns="http://www.w3.org/2000/svg"
                fill="none"
                viewBox="0 0 24 24"
                strokeWidth="1.5"
                stroke="currentColor"
                className="w-8 h-8 cursor-pointer"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M21 21l-5.197-5.197m0 0A7.5 7.5 0 105.196 5.196a7.5 7.5 0 0010.607 10.607z"
                />
              </svg>
            </div>
          </div>
          {/* Checkboxes */}
          <div className="flex flex-wrap items-center justify-center">
            {checkBoxListState}
          </div>
          {/* selected: {selectedTags} */}
          <FoundSnippetsCounter />
        </div>
      </div>
    </>
  );
};

export default SearchBar;
