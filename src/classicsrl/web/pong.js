// Scores belong to the image BEFORE each recorded action, not its resulting reward.
export function pongScores(decisions) {
  const scores = [{ agent: 0, opponent: 0 }];
  for (const [index, decision] of decisions.entries()) {
    if (decision.frame_index !== index || ![-1, 0, 1].includes(decision.reward)) return [];
    const previous = scores.at(-1);
    scores.push({ agent: previous.agent + Math.max(0, decision.reward),
      opponent: previous.opponent + Math.max(0, -decision.reward) });
  }
  return scores;
}

const original = [[144, 72, 17], [213, 130, 74], [92, 186, 92], [236, 236, 236]];
const minimal = [[12, 12, 12], [160, 160, 160], [242, 242, 242], [255, 255, 255]];

export function recolorPong(image) {
  const pixels = image.data;
  for (let offset = 0; offset < pixels.length; offset += 4) {
    let nearest = 0, distance = Infinity;
    for (let color = 0; color < original.length; color++) {
      const [red, green, blue] = original[color];
      const candidate = (pixels[offset] - red) ** 2 + (pixels[offset + 1] - green) ** 2
        + (pixels[offset + 2] - blue) ** 2;
      if (candidate < distance) { nearest = color; distance = candidate; }
    }
    const x = offset / 4 % image.width, y = Math.floor(offset / 4 / image.width);
    const color = nearest === 0 && x === 79 && y % 10 < 5 ? [48, 48, 48] : minimal[nearest];
    pixels[offset] = color[0]; pixels[offset + 1] = color[1]; pixels[offset + 2] = color[2];
    pixels[offset + 3] = 255;
  }
  return image;
}

export function pongRenderer(video, canvas) {
  const capture = document.createElement('canvas');
  capture.width = 160; capture.height = 210;
  const source = capture.getContext('2d', { willReadFrequently: true });
  const output = canvas.getContext('2d');
  return () => {
    if (video.readyState < 2 || video.videoWidth !== 160 || video.videoHeight !== 210) return;
    source.drawImage(video, 0, 0);
    // ALE Pong's square court lies between the score/header and the bottom border.
    output.putImageData(recolorPong(source.getImageData(0, 34, 160, 160)), 0, 0);
  };
}
