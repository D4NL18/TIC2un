import { TestBed } from '@angular/core/testing';

import { CnnFinetunningService } from './cnn-finetunning.service';

describe('CnnFinetunningService', () => {
  let service: CnnFinetunningService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(CnnFinetunningService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
