import os
import os.path
import cv2
import glob
import imutils


def main():
    counts = {}
    captcha_image_files = glob.glob(os.path.join("./img", "*"))
    for filename in captcha_image_files:
        image = cv2.imread("./{}".format(filename))

        # 图片转为灰度图 二维数组
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 在图像周围增加一些额外的填充区域(四周增加8像素)
        # gray = cv2.copyMakeBorder(gray, 8, 8, 8, 8, cv2.BORDER_REPLICATE)

        # 阈值图像（将其转换为纯黑色和白色）
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        # 找到图像轮廓（像素的连续斑点/阴影区域）
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 用于兼容不同的OPENCV版本，取出指定连续区域的坐标
        contours = contours[0] if imutils.is_cv2() else contours[1]

        # 字母图像区域
        letter_image_regions = []

        # 现在我们可以遍历四个轮廓中的每一个，并提取每个轮廓中的字母
        for contour in contours:
            # Get the rectangle that contains the contour
            (x, y, w, h) = cv2.boundingRect(contour)

            # Compare the width and height of the contour to detect letters that
            # are conjoined into one chunk
            if w / h > 1.25:
                # This contour is too wide to be a single letter!
                # Split it in half into two letter regions!
                half_width = int(w / 2)
                letter_image_regions.append((x, y, half_width, h))
                letter_image_regions.append((x + half_width, y, half_width, h))
            else:
                # This is a normal letter by itself
                letter_image_regions.append((x, y, w, h))

            print(letter_image_regions)
            # cv2.imshow("Image", contours)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()
            # return

        # 如果我们在验证码中发现多于或少于4个字母，我们的字母提取不能正确工作。 跳过图片而不是保存糟糕的训练数据！
        if len(letter_image_regions) != 4:
            continue
        else:
            print(filename)

        # 根据x坐标对检测到的字母图像进行排序，以确保我们正在从左到右地处理它们，以便我们将正确的图像与正确的字母相匹配
        letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

        # 保存每个字母的单个图像
        for letter_bounding_box, letter_text in zip(letter_image_regions, [str(i) for i in range(len(letter_image_regions))]):
            # Grab the coordinates of the letter in the image
            x, y, w, h = letter_bounding_box

            # Extract the letter from the original image with a 2-pixel margin around the edge
            letter_image = gray[y - 2:y + h + 2, x - 2:x + w + 2]

            # Get the folder to save the image in
            save_path = os.path.join("./", letter_text)

            # if the output directory does not exist, create it
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # write the letter image to a file
            count = counts.get(letter_text, 1)
            p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
            cv2.imwrite(p, letter_image)

            # increment the count for the current key
            counts[letter_text] = count + 1
        return


if __name__ == '__main__':
    main()