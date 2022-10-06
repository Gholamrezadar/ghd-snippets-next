import { ChangeEvent } from 'react';

const CheckBoxItem = ({
  tag,
  onCheck,
}: {
  tag: string;
  onCheck: (checked: boolean) => void;
}) => {
  return (
    <>
      <div className="mx-5">
        <label
          className="mx-1.5 ml-2 text-md  text-gray-900 dark:text-gray-300 cursor-pointer"
          htmlFor={tag}
        >
          {tag}
        </label>
        <input
          className="appearance-none w-[1.125rem] h-[1.125rem] text-blue-600 rounded-[4px] dark:checked:bg-blue-600 focus:outline-none transition duration-200 bg-no-repeat bg-center bg-contain cursor-pointer align-middle  dark:bg-ghd-dark-checkbox-disabled"
          type="checkbox"
          id={tag}
          onChange={(e: ChangeEvent<HTMLInputElement>) =>
            onCheck(e.target.checked)
          }
        />
      </div>
    </>
  );
};

export default CheckBoxItem;
