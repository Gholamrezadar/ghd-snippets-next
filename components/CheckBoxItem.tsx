import { ChangeEvent } from 'react';
import useStore from '../lib/store';

const CheckBoxItem = ({ tag }: { tag: string }) => {
  console.log(`checkbox-${tag} rerender`);
  const updateSelectedTags = useStore((state) => state.updateSelectedTags);
  return (
    <>
      <div className="md:mx-5 mx-1">
        <label
          className="mx-1.5 text-md text-gray-900 dark:text-gray-300 cursor-pointer"
          htmlFor={tag}
        >
          {tag}
        </label>
        <input
          className="appearance-none w-[1.125rem] h-[1.125rem] text-blue-600 rounded-[4px] dark:checked:bg-blue-600 focus:outline-none transition duration-200 ease-out bg-no-repeat bg-center bg-contain cursor-pointer align-middle dark:bg-ghd-dark-checkbox-disabled dark:checked:rotate-45"
          type="checkbox"
          id={tag}
          onChange={(e: ChangeEvent<HTMLInputElement>) => {
            const checked = e.target.checked;
            if (checked) {
              updateSelectedTags(tag, true); // => setSelectedTags([...selectedTags, tag]);
            }
            if (!checked) {
              updateSelectedTags(tag, false); // => setSelectedTags(selectedTags.filter((value) => value !== tag));
            }
          }}
        />
      </div>
    </>
  );
};

export default CheckBoxItem;
